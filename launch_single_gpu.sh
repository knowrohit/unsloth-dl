#!/bin/bash
# Launch script for single-GPU training

echo "=================================="
echo "Single GPU Training Launcher"
echo "=================================="

# Check GPU availability
echo ""
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch training
echo ""
echo "Launching training on GPU 0..."
echo ""

python3 train_single_gpu.py

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="

