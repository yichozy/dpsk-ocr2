#!/bin/bash
set -e

echo "Starting installation..."

# 1. Install PyTorch (CUDA 11.8)
echo "Installing PyTorch ecosystem..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 2. Install vLLM
echo "Installing vLLM..."
# Using direct URL for vLLM 0.8.5 wheel compatible with CUDA 11.8
pip install https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# 3. Install other requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 4. Install flash-attn
echo "Installing flash-attn..."
pip install flash-attn==2.7.3 --no-build-isolation

echo "Installation completed successfully."
