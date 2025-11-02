#!/bin/bash
# Alternative launcher using Accelerate
# This is simpler but may require initial configuration

echo "=================================="
echo "Accelerate Multi-GPU Launcher"
echo "=================================="

# First time setup (run this once):
# accelerate config

# Launch training
accelerate launch train_distributed.py

echo "=================================="
echo "Training completed!"
echo "=================================="

