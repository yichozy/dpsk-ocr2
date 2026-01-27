# DeepSeek-OCR-2 API Service

This repository provides an API service and Docker support for [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2), utilizing the V2 model with "Visual Causal Flow". This is a port of the API structure from [dpsk-ocr](https://github.com/yichozy/dpsk-ocr) adapted for the newer model architecture.

## Features

- **DeepSeek-OCR-2 Model**: Uses the latest V2 architecture for improved OCR performance and layout analysis.
- **FastAPI Backend**: Efficient, async-capable API for PDF and image processing.
- **Dockerized**: Ready-to-deploy Docker image with CUDA 11.8, PyTorch 2.6, and vLLM 0.8.5 optimization.
- **Task Queue**: Background processing for handling large PDF files without blocking the API.
- **Layout Detection**: Returns structured markdown, detection coordinates, and layout visualizations.

## Prerequisites

- NVIDIA GPU (Ampere or newer recommended, e.g., A100, A10, RTX 3090/4090)
- roughly 16GB+ VRAM for the model.
- Linux with CUDA drivers installed.

## Installation

### Option 1: Docker (Recommended)

1.  **Build the image:**
    ```bash
    docker build -t deepseek-ocr2 .
    ```

2.  **Run the container:**
    ```bash
    docker run --gpus all -p 8000:8000 -v $(pwd)/data:/app/data deepseek-ocr2
    ```

### Option 2: Local Conda Environment

1.  **Create Environment:**
    ```bash
    conda create -n deepseek-ocr2 python=3.12 -y
    conda activate deepseek-ocr2
    ```

2.  **Install GPU Dependencies:**
    ```bash
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
    
    # Install vLLM (Compatible wheel for Python 3.12/CUDA 11.8)
    wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
    pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
    
    pip install flash-attn==2.7.3 --no-build-isolation
    ```

3.  **Install Python Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the API:**
    ```bash
    python serve_pdf.py
    ```

## Usage

### API Endpoints

The API runs at `http://localhost:8000`. Full interactive documentation is available at `http://localhost:8000/docs`.

#### 1. Process a PDF
**POST** `/process_pdf`
- Upload a PDF file to start processing.
- Returns a `job_id`.

#### 2. Check Status
**GET** `/result/{job_id}/status`
- Check if the job is `pending`, `processing`, `completed`, or `failed`.

#### 3. Get Results
- **Markdown**: `GET /result/{job_id}/markdown`
- **Markdown with Coordinates**: `GET /result/{job_id}/markdown_det`
- **Layout Visualization (PDF)**: `GET /result/{job_id}/layout_pdf`
- **Extracted Images**: `GET /result/{job_id}/images`

### CLI Scripts

You can also run the original scripts for direct processing:

- **Single Image**: `python run_dpsk_ocr2_image.py`
- **PDF**: `python run_dpsk_ocr2_pdf.py`
- **Batch Eval**: `python run_dpsk_ocr2_eval_batch.py`

*Note: Configure `config.py` to set input/output paths for CLI scripts.*

## Project Structure

- `serve_pdf.py`: Main API application.
- `DeepSeek-OCR2-vllm/`: Core model code.
- `config.py`: Configuration settings.
- `database.py` & `task_queue.py`: Job management.
- `pdf_utils.py` & `processing_utils.py`: Helper functions for image/PDF handling.

## License

This project follows the license of the original [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2) repository.
