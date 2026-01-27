# Use NVIDIA CUDA 11.8 devel image
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Create working directory
WORKDIR /app

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Download and install vllm wheel (cp38-abi3 which is compatible with python 3.10+)
RUN wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl -O vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl && \
    pip install --no-cache-dir wheel packaging ninja && \
    pip install --no-cache-dir vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl && \
    rm vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Copy requirements if exists (referencing remote one since local might be separate)
# For this dockerfile we just list the deps directly or copy local requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --ignore-installed dbus-python python-apt || true

# Install additional packages for API
RUN pip install --no-cache-dir fastapi uvicorn PyMuPDF img2pdf easydict addict python-dotenv

# Install flash-attn
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p tmp/pdf_ocr

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "serve_pdf.py"]
