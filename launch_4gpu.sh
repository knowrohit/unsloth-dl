#!/bin/bash
# Launch script for 4-GPU distributed training with Unsloth

echo "=================================="
echo "Multi-GPU Training Launcher"
echo "Using 4x RTX 5070 GPUs"
echo "=================================="

# Check GPU availability
echo ""
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL Configuration for multi-GPU communication
export NCCL_DEBUG=ERROR  # Minimal logging to reduce overhead
export NCCL_IB_DISABLE=1  # Disable InfiniBand (not needed for single node)
# Let NCCL auto-detect network interface (remove SOCKET_IFNAME)
export NCCL_P2P_DISABLE=0  # Try enabling P2P first
export NCCL_SHM_DISABLE=0  # Enable shared memory

# Increase timeouts for slow initialization
export NCCL_TIMEOUT=1800  # 30 minutes timeout (default is 10 minutes)
export NCCL_BLOCKING_WAIT=1  # Blocking wait instead of polling

# Network optimizations
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS_PER_PEER=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Alternative: If P2P doesn't work, try disabling it
# Uncomment the next line if you still get NCCL errors:
# export NCCL_P2P_DISABLE=1

# Launch distributed training with torchrun
echo ""
echo "Launching training on 4 GPUs..."
echo ""

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_distributed.py

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="

