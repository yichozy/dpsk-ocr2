#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting installation script for DeepSeek OCR 2..."

# 1. Install PyTorch depending on OS/Hardware
OS="$(uname)"
ARCH="$(uname -m)"

echo "Detected OS: $OS, Arch: $ARCH"

if [[ "$OS" == "Darwin" ]]; then
    # MacOS
    echo "Installing PyTorch for MacOS (MPS support if available)..."
    pip install torch torchvision torchaudio
    
    echo "Note: vllm and flash-attn on MacOS are experimental or require source build. Skipping pre-built wheel installation."
    
elif [[ "$OS" == "Linux" ]]; then
    # Linux
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected. Installing PyTorch with CUDA 11.8 support..."
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
        
        echo "Installing vLLM..."
        # Using curl for better portability, falling back to wget
        VLLM_WHL="vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl"
        VLLM_URL="https://github.com/vllm-project/vllm/releases/download/v0.8.5/$VLLM_WHL"
        
        if command -v curl &> /dev/null; then
            curl -L -o "$VLLM_WHL" "$VLLM_URL"
        elif command -v wget &> /dev/null; then
            wget -O "$VLLM_WHL" "$VLLM_URL"
        else
            echo "Error: Neither curl nor wget found. Cannot download vLLM."
            exit 1
        fi
        
        pip install wheel packaging ninja
        pip install "$VLLM_WHL"
        rm "$VLLM_WHL"
        
        echo "Installing Flash Attention..."
        pip install flash-attn==2.7.3 --no-build-isolation
    else
        echo "No NVIDIA GPU detected. Installing CPU versions..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# 2. Install general requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# 3. Install additional dependencies if not already in requirements
# (The Dockerfile lists these explicitly, checking them ensures they are present)
echo "Ensuring API dependencies are installed..."
pip install fastapi uvicorn PyMuPDF img2pdf easydict addict python-dotenv matplotlib

echo "Installation complete!"
