#!/bin/bash
# Launch script for 4-GPU distributed training (WITHOUT P2P)
# Use this if launch_4gpu.sh fails with NCCL errors

echo "=================================="
echo "Multi-GPU Training Launcher (No P2P)"
echo "Using 4x RTX 5070 GPUs"
echo "=================================="

# Check GPU availability
echo ""
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL Configuration - DISABLE P2P (slower but more reliable)
export NCCL_DEBUG=ERROR
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1  # DISABLE peer-to-peer (use CPU/socket instead)
export NCCL_SHM_DISABLE=0

# Increase timeouts
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch distributed training with torchrun
echo ""
echo "Launching training on 4 GPUs (without P2P)..."
echo ""

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_distributed.py

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="

