#!/bin/bash
# Run configurations 5-8 in parallel, one on each GPU

echo "=================================="
echo "Parallel Training: Configs 5-8"
echo "Running 4 configurations simultaneously"
echo "=================================="

# Create output directories
mkdir -p outputs
mkdir -p models
mkdir -p logs

# Check GPU availability
echo ""
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Weights & Biases project name
WANDB_PROJECT="math-verification-multiconfig"

echo ""
echo "=================================="
echo "Starting 4 parallel training runs (Configs 5-8)..."
echo "W&B Project: $WANDB_PROJECT"
echo "=================================="
echo ""

# Configuration 5: Very low rank (r=1), max_steps=60, lora_alpha=2
echo "Config 5 (GPU 0): LoRA r=1, alpha=2, LR=2e-4, max_steps=60"
CUDA_VISIBLE_DEVICES=0 python3 train_single_gpu_configurable.py \
    --gpu_id 0 \
    --lora_r 1 \
    --lora_alpha 2 \
    --learning_rate 2e-4 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_seq_length 1024 \
    --train_samples 10000 \
    --max_steps 60 \
    --wandb_project "$WANDB_PROJECT" \
    --output_name "config5_r1_alpha2_lr2e4_maxsteps60" \
    > logs/config5.log 2>&1 &

PID5=$!
echo "  Started Config 5 (PID: $PID5)"

# Configuration 6: Low rank r=4, alpha=8, epochs
echo "Config 6 (GPU 1): LoRA r=4, alpha=8, LR=2e-4, epochs=3"
CUDA_VISIBLE_DEVICES=1 python3 train_single_gpu_configurable.py \
    --gpu_id 1 \
    --lora_r 4 \
    --lora_alpha 8 \
    --learning_rate 2e-4 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_seq_length 1024 \
    --train_samples 10000 \
    --epochs 3 \
    --wandb_project "$WANDB_PROJECT" \
    --output_name "config6_r4_alpha8_lr2e4" \
    > logs/config6.log 2>&1 &

PID6=$!
echo "  Started Config 6 (PID: $PID6)"

# Configuration 7: Very high rank r=64, alpha=128, epochs
echo "Config 7 (GPU 2): LoRA r=64, alpha=128, LR=2e-4, epochs=3"
CUDA_VISIBLE_DEVICES=2 python3 train_single_gpu_configurable.py \
    --gpu_id 2 \
    --lora_r 64 \
    --lora_alpha 128 \
    --learning_rate 2e-4 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_seq_length 1024 \
    --train_samples 10000 \
    --epochs 3 \
    --wandb_project "$WANDB_PROJECT" \
    --output_name "config7_r64_alpha128_lr2e4" \
    > logs/config7.log 2>&1 &

PID7=$!
echo "  Started Config 7 (PID: $PID7)"

# Configuration 8: Low rank r=1, alpha=4, higher LR, epochs
echo "Config 8 (GPU 3): LoRA r=1, alpha=4, LR=3e-4, epochs=3"
CUDA_VISIBLE_DEVICES=3 python3 train_single_gpu_configurable.py \
    --gpu_id 3 \
    --lora_r 1 \
    --lora_alpha 4 \
    --learning_rate 3e-4 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_seq_length 1024 \
    --train_samples 10000 \
    --epochs 3 \
    --wandb_project "$WANDB_PROJECT" \
    --output_name "config8_r1_alpha4_lr3e4" \
    > logs/config8.log 2>&1 &

PID8=$!
echo "  Started Config 8 (PID: $PID8)"

echo ""
echo "=================================="
echo "All 4 training jobs launched!"
echo "=================================="
echo ""
echo "Process IDs:"
echo "  Config 5 (GPU 0): $PID5"
echo "  Config 6 (GPU 1): $PID6"
echo "  Config 7 (GPU 2): $PID7"
echo "  Config 8 (GPU 3): $PID8"
echo ""
echo "Monitor logs with:"
echo "  tail -f logs/config5.log"
echo "  tail -f logs/config6.log"
echo "  tail -f logs/config7.log"
echo "  tail -f logs/config8.log"
echo ""
echo "Monitor GPU usage with:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Waiting for all training jobs to complete..."
echo ""

# Wait for all background jobs
wait $PID5
STATUS5=$?
echo "Config 5 completed (exit code: $STATUS5)"

wait $PID6
STATUS6=$?
echo "Config 6 completed (exit code: $STATUS6)"

wait $PID7
STATUS7=$?
echo "Config 7 completed (exit code: $STATUS7)"

wait $PID8
STATUS8=$?
echo "Config 8 completed (exit code: $STATUS8)"

echo ""
echo "=================================="
echo "All training jobs completed!"
echo "=================================="
echo ""
echo "Results:"
echo "  Config 5: Exit code $STATUS5 - models/config5_r1_alpha2_lr2e4_maxsteps60"
echo "  Config 6: Exit code $STATUS6 - models/config6_r4_alpha8_lr2e4"
echo "  Config 7: Exit code $STATUS7 - models/config7_r64_alpha128_lr2e4"
echo "  Config 8: Exit code $STATUS8 - models/config8_r1_alpha4_lr3e4"
echo ""
echo "Logs saved in logs/ directory"
echo "Models saved in models/ directory"
echo ""

