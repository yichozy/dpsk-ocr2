import os
import io
import uuid
import shutil
import torch
import hashlib
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

from pdf_utils import pdf_to_images_high_quality
from processing_utils import (
    pil_to_pdf_img2pdf,
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

# Environment setup
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from config import MODEL_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

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

# Initialize database and queue on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and task queue on application startup"""
    global task_queue
    init_database()
    # Initialize queue with 1 worker for sequential processing
    task_queue = get_queue(max_workers=1)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    shutdown_queue()

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
    max_model_len=8192,
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
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)

# Create directories for temporary files
TEMP_DIR = Path("tmp/pdf_ocr")
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
        images = pdf_to_images_high_quality(pdf_path)

        # Update total pages
        update_task_status(job_id, "processing", total_pages=len(images))

        # Preprocess images in parallel
        prompt = PROMPT
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_inputs = list(executor.map(
                lambda img: process_single_image(img, prompt),
                images
            ))

        # Run OCR inference
        outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

        # Process outputs
        contents_det = ''
        contents = ''
        draw_images = []

        for jdx, (output, img) in enumerate(zip(outputs_list, images)):
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
            draw_images.append(result_image)

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

        # Save outputs
        mmd_det_path = output_dir / "output_det.mmd"
        mmd_path = output_dir / "output.mmd"
        pdf_out_path = output_dir / "output_layouts.pdf"

        with open(mmd_det_path, 'w', encoding='utf-8') as f:
            f.write(contents_det)

        with open(mmd_path, 'w', encoding='utf-8') as f:
            f.write(contents)

        pil_to_pdf_img2pdf(draw_images, str(pdf_out_path))

        # Update status to completed
        update_task_status(job_id, "completed", processed_pages=len(images))

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
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}


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

    # Check for existing completed task with same hash
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
