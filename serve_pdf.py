import os
import io
import uuid
import shutil
import torch
import hashlib
import signal
import sys
import tempfile
import tempfile
import logging
import threading
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

from pdf_utils import pdf_to_images_high_quality, get_pdf_page_count, get_pdf_page_batch
from processing_utils import (
    pil_to_pdf_img2pdf,
    paths_to_pdf_img2pdf,
    re_match,
    process_image_with_refs
)
from database import (
    init_database,
    create_task,
    update_task_status,
    get_task_status,
    delete_task,
    delete_task,
    get_all_tasks,
    get_completed_task_by_hash
)
from task_queue import get_queue, shutdown_queue
from memory_monitor import init_monitor, stop_monitor, MemoryStats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment setup
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from config import (
    MODEL_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE, PDF_BATCH_SIZE,
    MEMORY_WARNING_THRESHOLD, MEMORY_CRITICAL_THRESHOLD, MEMORY_CHECK_INTERVAL, MAX_PDF_WORKERS, ENABLE_CACHE
)

from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCR2Processor

# Register custom model
ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek OCR PDF Service",
    description="OCR service for PDF documents with layout detection",
    version="1.0.0"
)

# Initialize task queue (global instance)
task_queue = None
_shutdown_requested = False


def handle_oom_critical(stats: MemoryStats):
    """
    Handler for critical OOM situations.
    Initiates graceful shutdown to prevent crashes.
    """
    global _shutdown_requested

    if _shutdown_requested:
        return  # Already handling shutdown

    _shutdown_requested = True
    logger.critical(
        f"CRITICAL OOM DETECTED: {stats.percent_used:.1f}% memory used. "
        f"Initiating graceful shutdown..."
    )

    # Give some time for current requests to complete
    import threading
    def delayed_shutdown():
        logger.warning("Service will restart in 10 seconds due to OOM...")
        threading.Event().wait(10)
        logger.critical("Initiating restart due to OOM")
        # Exit with code that will trigger a container restart
        sys.exit(137)  # 137 = 128 + 9 (SIGKILL), typical Docker OOM exit code

    shutdown_thread = threading.Thread(target=delayed_shutdown, daemon=True)
    shutdown_thread.start()


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        global _shutdown_requested
        _shutdown_requested = True
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


# Initialize database and queue on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database, task queue, and memory monitor on startup"""
    global task_queue

    setup_signal_handlers()

    init_database()
    # Initialize queue for checking number of max workers
    task_queue = get_queue(max_workers=MAX_PDF_WORKERS)

    # Initialize memory monitor with OOM handler
    try:
        init_monitor(
            warning_threshold=MEMORY_WARNING_THRESHOLD,
            critical_threshold=MEMORY_CRITICAL_THRESHOLD,
            check_interval=MEMORY_CHECK_INTERVAL,
            on_critical=handle_oom_critical
        )
        logger.info("Memory monitor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize memory monitor: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down service...")
    shutdown_queue()
    stop_monitor()
    logger.info("Shutdown complete")

# Security setup
security = HTTPBearer()
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the authentication token"""
    if AUTH_TOKEN is None:
        # If no AUTH_TOKEN is set, allow access without authentication
        return True

    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# Initialize model (loaded once at startup)
llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"], "model_type": "deepseek_v2"},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True,
    max_model_len=16384,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    disable_mm_preprocessor_cache=True
)

# Setup logits processors and sampling params
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=20,
        window_size=50,
        whitelist_token_ids={128821, 128822}  # <td>, </td>
    )
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=15000,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)

# Global lock to prevent concurrent VLLM offline inference crashes
llm_lock = threading.Lock()

# Create directories for temporary files
TEMP_DIR = Path(tempfile.gettempdir()) / "pdf_ocr"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class ProcessingStatus(BaseModel):
    """Response model for processing status"""
    job_id: str
    status: str
    message: Optional[str] = None


class OCRResult(BaseModel):
    """Response model for OCR results"""
    job_id: str
    markdown_content: str
    markdown_with_det: str
    layout_pdf_url: str
    extracted_images: list[str]


def process_single_image(image, prompt):
    """Prepare single image for batch processing"""
    cache_item = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": DeepseekOCR2Processor().tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=CROP_MODE
            )
        },
    }
    return cache_item


def cleanup_job_files(job_id: str):
    """Clean up temporary files for a job"""
    job_dir = TEMP_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)


def process_pdf_background(pdf_path: str, job_id: str, output_dir: Path):
    """
    Background task wrapper for PDF processing

    Args:
        pdf_path: Path to input PDF file
        job_id: Unique job identifier
        output_dir: Directory to save outputs
    """
    try:
        process_pdf_internal(pdf_path, job_id, output_dir)
    except Exception as e:
        # Ensure status is updated even if something goes wrong
        update_task_status(job_id, "failed", error_message=str(e))


def process_pdf_internal(pdf_path: str, job_id: str, output_dir: Path):
    """
    Internal function to process PDF

    Processes PDF pages in batches to limit memory usage and avoid system memory leaks.

    Args:
        pdf_path: Path to input PDF file
        job_id: Unique job identifier
        output_dir: Directory to save outputs
    """
    try:
        # Update status to processing
        update_task_status(job_id, "processing")

        # Create output directories
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Convert PDF to images
        total_pages = get_pdf_page_count(pdf_path)

        # Update total pages
        update_task_status(job_id, "processing", total_pages=total_pages)

        # Initialize output accumulators
        contents_det = ''
        contents = ''
        draw_image_paths = []

        # Process in batches to limit memory usage
        # Use configured batch size (adjust in config.py based on available system memory)
        prompt = PROMPT

        for batch_start in range(0, total_pages, PDF_BATCH_SIZE):
            batch_end = min(batch_start + PDF_BATCH_SIZE, total_pages)
            batch_images = get_pdf_page_batch(pdf_path, batch_start, batch_end)

            # Preprocess this batch in parallel
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                batch_inputs = list(executor.map(
                    lambda img: process_single_image(img, prompt),
                    batch_images
                ))

            # Run OCR inference on this batch (vllm offline LLM is not thread-safe)
            with llm_lock:
                outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

            # Process outputs for this batch
            for batch_idx, (output, img) in enumerate(zip(outputs_list, batch_images)):
                jdx = batch_start + batch_idx
                content = output.outputs[0].text

                # Check for proper completion
                if '<｜end▁of▁sentence｜>' in content:
                    content = content.replace('<｜end▁of▁sentence｜>', '')
                else:
                    if SKIP_REPEAT:
                        continue

                page_num = f'\n<--- Page Split --->'
                contents_det += content + f'\n{page_num}\n'

                # Extract layout references
                matches_ref, matches_images, mathes_other = re_match(content)

                # Draw bounding boxes and extract images
                image_draw = img.copy()
                result_image = process_image_with_refs(
                    image_draw,
                    matches_ref,
                    jdx,
                    str(output_dir)
                )
                
                drawn_img_path = str(output_dir / f"drawn_{jdx}.jpg")
                if result_image.mode != 'RGB':
                    result_image = result_image.convert('RGB')
                result_image.save(drawn_img_path, format="JPEG", quality=95)
                draw_image_paths.append(drawn_img_path)

                # clean up references
                del result_image
                del image_draw

                # Replace image references with markdown links
                for idx, a_match_image in enumerate(matches_images):
                    content = content.replace(
                        a_match_image,
                        f'![](images/{jdx}_{idx}.jpg)\n'
                    )

                # Clean up other references
                for idx, a_match_other in enumerate(mathes_other):
                    content = content.replace(a_match_other, '') \
                        .replace('\\coloneqq', ':=') \
                        .replace('\\eqqcolon', '=:') \
                        .replace('\n\n\n\n', '\n\n') \
                        .replace('\n\n\n', '\n\n')

                contents += content + f'\n{page_num}\n'

            # Explicitly clean up batch tensors to free system memory
            del batch_inputs, outputs_list, batch_images
            import gc
            gc.collect()

        # Save outputs
        mmd_det_path = output_dir / "output_det.mmd"
        mmd_path = output_dir / "output.mmd"
        pdf_out_path = output_dir / "output_layouts.pdf"

        with open(mmd_det_path, 'w', encoding='utf-8') as f:
            f.write(contents_det)

        with open(mmd_path, 'w', encoding='utf-8') as f:
            f.write(contents)

        paths_to_pdf_img2pdf(draw_image_paths, str(pdf_out_path))

        for p in draw_image_paths:
            try:
                import os
                os.remove(p)
            except:
                pass

        # Update status to completed
        update_task_status(job_id, "completed", processed_pages=total_pages)

        return {
            "success": True,
            "markdown_path": str(mmd_path),
            "markdown_det_path": str(mmd_det_path),
            "layout_pdf_path": str(pdf_out_path),
            "images_dir": str(images_dir)
        }

    except Exception as e:
        # Update status to failed
        update_task_status(job_id, "failed", error_message=str(e))

        return {
            "success": False,
            "error": str(e)
        }


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "DeepSeek OCR PDF Service",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint with detailed memory statistics.

    Returns:
        Health status including:
        - status: overall health status
        - model_loaded: whether the model is loaded
        - memory: detailed memory statistics
        - alerts: any active memory alerts
    """
    from memory_monitor import get_monitor

    health_data = {
        "status": "healthy",
        "model_loaded": True,
        "memory": {},
        "alerts": []
    }

    # Get memory statistics if monitor is running
    monitor = get_monitor()
    if monitor:
        try:
            stats = monitor.get_memory_stats()
            health_data["memory"] = {
                "system": {
                    "total_mb": round(stats.total_mb, 2),
                    "used_mb": round(stats.used_mb, 2),
                    "available_mb": round(stats.available_mb, 2),
                    "percent_used": round(stats.percent_used, 2)
                }
            }

            # Add GPU memory if available
            if stats.gpu_total_mb is not None:
                health_data["memory"]["gpu"] = {
                    "total_mb": round(stats.gpu_total_mb, 2),
                    "used_mb": round(stats.gpu_used_mb, 2),
                    "free_mb": round(stats.gpu_free_mb, 2)
                }

            # Check for alerts
            if stats.percent_used >= 90:
                health_data["status"] = "critical"
                health_data["alerts"].append("CRITICAL: System memory usage above 90%")
            elif stats.percent_used >= 80:
                health_data["status"] = "warning"
                health_data["alerts"].append("WARNING: System memory usage above 80%")

        except Exception as e:
            health_data["status"] = "degraded"
            health_data["alerts"].append(f"Memory monitoring error: {str(e)}")
    else:
        health_data["alerts"].append("Memory monitor not available")

    # Add queue status
    if task_queue:
        queue_size = task_queue.get_queue_size()
        all_tasks = task_queue.get_all_tasks()
        health_data["queue"] = {
            "size": queue_size,
            "active_tasks": len(all_tasks)
        }

    return health_data


@app.get("/queue/status")
async def queue_status(authenticated: bool = Depends(verify_token)):
    """Get current queue status"""
    queue_size = task_queue.get_queue_size()
    all_tasks = task_queue.get_all_tasks()

    # Separate tasks by status
    queued = [tid for tid, info in all_tasks.items() if info['status'] == 'queued']
    processing = [tid for tid, info in all_tasks.items() if info['status'] == 'processing']

    return {
        "queue_size": queue_size,
        "total_tasks": len(all_tasks),
        "queued_tasks": len(queued),
        "processing_tasks": len(processing),
        "queued_task_ids": queued,
        "processing_task_ids": processing
    }


@app.post("/process_pdf", response_model=ProcessingStatus)
async def process_pdf(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_token)
):
    """
    Upload and process a PDF file

    Args:
        file: PDF file to process

    Returns:
        Job ID and status (processing initiated)
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file content first to compute hash
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

    # Generate hash for the file content
    file_hash = hashlib.sha256(contents).hexdigest()

    # Check for existing completed task if caching is enabled
    if ENABLE_CACHE:
        existing_task = get_completed_task_by_hash(file_hash)
        if existing_task:
            # Check if the result file actually exists
            existing_mmd_path = TEMP_DIR / existing_task['job_id'] / "output.mmd"
            if existing_mmd_path.exists():
                return ProcessingStatus(
                    job_id=existing_task['job_id'],
                    status="completed",
                    message="Retrieved from cache (file already processed)"
                )

    # Generate unique job ID for new task
    job_id = str(uuid.uuid4())

    # Create job directory
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    # Save uploaded file
    pdf_path = job_dir / "input.pdf"
    try:
        with open(pdf_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        # Cleanup if save fails
        cleanup_job_files(job_id)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create task in database
    create_task(job_id, file.filename, file_hash)

    # Add task to queue for sequential processing
    task_queue.add_task(job_id, process_pdf_background, str(pdf_path), job_id, job_dir)

    # Get current queue size
    queue_size = task_queue.get_queue_size()

    # Return immediately with pending status
    return ProcessingStatus(
        job_id=job_id,
        status="pending",
        message=f"PDF processing queued (position: {queue_size}). Use /result/{job_id}/status to check progress."
    )


@app.get("/result/{job_id}/status")
async def get_status(job_id: str, authenticated: bool = Depends(verify_token)):
    """Get processing status for a job"""
    task = get_task_status(job_id)

    if not task:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": task["job_id"],
        "status": task["status"],
        "filename": task["filename"],
        "total_pages": task["total_pages"],
        "processed_pages": task["processed_pages"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "error_message": task["error_message"]
    }


@app.get("/result/{job_id}/markdown")
async def get_markdown(job_id: str, authenticated: bool = Depends(verify_token)):
    """Get markdown output for a job"""
    # Check task status first
    task = get_task_status(job_id)
    if not task:
        raise HTTPException(status_code=404, detail="Job not found")

    if task["status"] == "pending":
        raise HTTPException(status_code=202, detail="Processing not started yet")
    elif task["status"] == "processing":
        raise HTTPException(status_code=202, detail="Processing in progress")
    elif task["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Processing failed: {task['error_message']}")

    mmd_path = TEMP_DIR / job_id / "output.mmd"

    if not mmd_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    with open(mmd_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return {"job_id": job_id, "status": "completed", "content": content}


@app.get("/result/{job_id}/markdown_det")
async def get_markdown_with_detection(job_id: str, authenticated: bool = Depends(verify_token)):
    """Get markdown with detection annotations for a job"""
    # Check task status first
    task = get_task_status(job_id)
    if not task:
        raise HTTPException(status_code=404, detail="Job not found")

    if task["status"] == "pending":
        raise HTTPException(status_code=202, detail="Processing not started yet")
    elif task["status"] == "processing":
        raise HTTPException(status_code=202, detail="Processing in progress")
    elif task["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Processing failed: {task['error_message']}")

    mmd_det_path = TEMP_DIR / job_id / "output_det.mmd"

    if not mmd_det_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    with open(mmd_det_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return {"job_id": job_id, "status": "completed", "content": content}


@app.get("/result/{job_id}/layout_pdf")
async def get_layout_pdf(job_id: str, authenticated: bool = Depends(verify_token)):
    """Download layout visualization PDF"""
    # Check task status first
    task = get_task_status(job_id)
    if not task:
        raise HTTPException(status_code=404, detail="Job not found")

    if task["status"] == "pending":
        raise HTTPException(status_code=202, detail="Processing not started yet")
    elif task["status"] == "processing":
        raise HTTPException(status_code=202, detail="Processing in progress")
    elif task["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Processing failed: {task['error_message']}")

    pdf_path = TEMP_DIR / job_id / "output_layouts.pdf"

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"layout_{job_id}.pdf"
    )


@app.get("/result/{job_id}/images")
async def list_extracted_images(job_id: str, authenticated: bool = Depends(verify_token)):
    """List all extracted images for a job"""
    # Check task status first
    task = get_task_status(job_id)
    if not task:
        raise HTTPException(status_code=404, detail="Job not found")

    if task["status"] == "pending":
        raise HTTPException(status_code=202, detail="Processing not started yet")
    elif task["status"] == "processing":
        raise HTTPException(status_code=202, detail="Processing in progress")
    elif task["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Processing failed: {task['error_message']}")

    images_dir = TEMP_DIR / job_id / "images"

    if not images_dir.exists():
        raise HTTPException(status_code=404, detail="Images not found")

    images = list(images_dir.glob("*.jpg"))
    return {
        "job_id": job_id,
        "status": "completed",
        "images": [img.name for img in images],
        "count": len(images)
    }


@app.get("/result/{job_id}/images/{image_name}")
async def get_extracted_image(job_id: str, image_name: str, authenticated: bool = Depends(verify_token)):
    """Download a specific extracted image"""
    # Check task status first
    task = get_task_status(job_id)
    if not task:
        raise HTTPException(status_code=404, detail="Job not found")

    if task["status"] == "pending":
        raise HTTPException(status_code=202, detail="Processing not started yet")
    elif task["status"] == "processing":
        raise HTTPException(status_code=202, detail="Processing in progress")
    elif task["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Processing failed: {task['error_message']}")

    image_path = TEMP_DIR / job_id / "images" / image_name

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        image_path,
        media_type="image/jpeg",
        filename=image_name
    )


@app.get("/tasks")
async def list_tasks(status: Optional[str] = None, authenticated: bool = Depends(verify_token)):
    """
    List all tasks, optionally filtered by status

    Args:
        status: Optional filter by status (pending, processing, completed, failed)

    Returns:
        List of all tasks
    """
    tasks = get_all_tasks(status)
    return {"tasks": tasks, "count": len(tasks)}


@app.delete("/result/{job_id}")
async def delete_job(job_id: str, authenticated: bool = Depends(verify_token)):
    """Delete all files and database entry associated with a job"""
    # Check if job exists
    task = get_task_status(job_id)
    if not task:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete files
    cleanup_job_files(job_id)

    # Delete from database
    delete_task(job_id)

    # Remove from queue tracking
    task_queue.remove_task_info(job_id)

    return {"job_id": job_id, "status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
