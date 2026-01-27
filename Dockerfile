# Use NVIDIA CUDA 11.8 devel image
# FROM --platform=linux/amd64 nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Python Virtual Environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Force CUDA installation in install.sh
ENV FORCE_CUDA=1

# Create working directory
WORKDIR /app

# Copy installation script and requirements first (for caching)
COPY install.sh .
COPY requirements.txt .

# Run installation script
# This handles system dependencies, venv creation, and python packages
RUN bash install.sh

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p tmp/pdf_ocr

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "serve_pdf.py"]
