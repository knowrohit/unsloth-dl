#!/bin/bash
# Run 8 different training configurations
# Configs 1-4 run in parallel (one on each GPU)
# Configs 5-8 run sequentially after all 4 complete (cycling through GPUs 0-3)

echo "=================================="
echo "Parallel Multi-Config Training"
echo "Running 8 different configurations"
echo "Configs 1-4: Parallel on GPUs 0-3"
echo "Configs 5-8: Sequential after all 4 complete"
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
echo "Starting 4 parallel training runs (Configs 1-4)..."
echo "W&B Project: $WANDB_PROJECT"
echo "=================================="
echo ""

# Configuration 1: High rank, standard learning rate
echo "Config 1 (GPU 0): LoRA r=32, LR=2e-4, seq=1024"
CUDA_VISIBLE_DEVICES=0 python3 train_single_gpu_configurable.py \
    --gpu_id 0 \
    --lora_r 32 \
    --learning_rate 2e-4 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_seq_length 1024 \
    --train_samples 10000 \
    --epochs 3 \
    --wandb_project "$WANDB_PROJECT" \
    --output_name "config1_r32_lr2e4" \
    > logs/config1.log 2>&1 &

PID1=$!
echo "  Started Config 1 (PID: $PID1)"

# Configuration 2: Medium rank, higher learning rate
echo "Config 2 (GPU 1): LoRA r=16, LR=3e-4, seq=1024"
CUDA_VISIBLE_DEVICES=1 python3 train_single_gpu_configurable.py \
    --gpu_id 1 \
    --lora_r 16 \
    --learning_rate 3e-4 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_seq_length 1024 \
    --train_samples 10000 \
    --epochs 3 \
    --wandb_project "$WANDB_PROJECT" \
    --output_name "config2_r16_lr3e4" \
    > logs/config2.log 2>&1 &

PID2=$!
echo "  Started Config 2 (PID: $PID2)"

# Configuration 3: Lower rank, standard LR, shorter sequences
echo "Config 3 (GPU 2): LoRA r=8, LR=2e-4, seq=512"
CUDA_VISIBLE_DEVICES=2 python3 train_single_gpu_configurable.py \
    --gpu_id 2 \
    --lora_r 8 \
    --learning_rate 2e-4 \
    --batch_size 4 \
    --grad_accum 2 \
    --max_seq_length 512 \
    --train_samples 10000 \
    --epochs 3 \
    --wandb_project "$WANDB_PROJECT" \
    --output_name "config3_r8_lr2e4_seq512" \
    > logs/config3.log 2>&1 &

PID3=$!
echo "  Started Config 3 (PID: $PID3)"

# Configuration 4: Medium rank, lower LR (more conservative)
echo "Config 4 (GPU 3): LoRA r=16, LR=1e-4, seq=1024"
CUDA_VISIBLE_DEVICES=3 python3 train_single_gpu_configurable.py \
    --gpu_id 3 \
    --lora_r 16 \
    --learning_rate 1e-4 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_seq_length 1024 \
    --train_samples 10000 \
    --epochs 3 \
    --wandb_project "$WANDB_PROJECT" \
    --output_name "config4_r16_lr1e4" \
    > logs/config4.log 2>&1 &

PID4=$!
echo "  Started Config 4 (PID: $PID4)"

echo ""
echo "=================================="
echo "All 4 training jobs launched!"
echo "=================================="
echo ""
echo "Process IDs:"
echo "  Config 1 (GPU 0): $PID1"
echo "  Config 2 (GPU 1): $PID2"
echo "  Config 3 (GPU 2): $PID3"
echo "  Config 4 (GPU 3): $PID4"
echo ""
echo "Monitor logs with:"
echo "  tail -f logs/config1.log"
echo "  tail -f logs/config2.log"
echo "  tail -f logs/config3.log"
echo "  tail -f logs/config4.log"
echo "  tail -f logs/config5.log  (after all 4 configs complete)"
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
wait $PID1
STATUS1=$?
echo "Config 1 completed (exit code: $STATUS1)"

wait $PID2
STATUS2=$?
echo "Config 2 completed (exit code: $STATUS2)"

wait $PID3
STATUS3=$?
echo "Config 3 completed (exit code: $STATUS3)"

wait $PID4
STATUS4=$?
echo "Config 4 completed (exit code: $STATUS4)"

echo ""
echo "=================================="
echo "First 4 training jobs completed!"
echo "=================================="
echo ""

# Configuration 5: Very low rank (r=1), max_steps=60, lora_alpha=2
echo "=================================="
echo "Starting Config 5 (GPU 0): LoRA r=1, alpha=2, LR=2e-4, max_steps=60"
echo "=================================="
echo ""
echo "Config 5 (GPU 0): LoRA r=1, alpha=2, LR=2e-4, max_steps=60, seq=1024"
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
    > logs/config5.log 2>&1

STATUS5=$?
echo "Config 5 completed (exit code: $STATUS5)"

# Configuration 6: Low rank r=4, alpha=8, epochs
echo ""
echo "=================================="
echo "Starting Config 6 (GPU 1): LoRA r=4, alpha=8, LR=2e-4, epochs=3"
echo "=================================="
echo ""
echo "Config 6 (GPU 1): LoRA r=4, alpha=8, LR=2e-4, epochs=3, seq=1024"
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
    > logs/config6.log 2>&1

STATUS6=$?
echo "Config 6 completed (exit code: $STATUS6)"

# Configuration 7: Very high rank r=64, alpha=128, epochs
echo ""
echo "=================================="
echo "Starting Config 7 (GPU 2): LoRA r=64, alpha=128, LR=2e-4, epochs=3"
echo "=================================="
echo ""
echo "Config 7 (GPU 2): LoRA r=64, alpha=128, LR=2e-4, epochs=3, seq=1024"
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
    > logs/config7.log 2>&1

STATUS7=$?
echo "Config 7 completed (exit code: $STATUS7)"

# Configuration 8: Low rank r=1, alpha=4, higher LR, epochs
echo ""
echo "=================================="
echo "Starting Config 8 (GPU 3): LoRA r=1, alpha=4, LR=3e-4, epochs=3"
echo "=================================="
echo ""
echo "Config 8 (GPU 3): LoRA r=1, alpha=4, LR=3e-4, epochs=3, seq=1024"
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
    > logs/config8.log 2>&1

STATUS8=$?
echo "Config 8 completed (exit code: $STATUS8)"

echo ""
echo "=================================="
echo "All 8 training jobs completed!"
echo "=================================="
echo ""
echo "Results:"
echo "  Config 1: Exit code $STATUS1 - models/config1_r32_lr2e4"
echo "  Config 2: Exit code $STATUS2 - models/config2_r16_lr3e4"
echo "  Config 3: Exit code $STATUS3 - models/config3_r8_lr2e4_seq512"
echo "  Config 4: Exit code $STATUS4 - models/config4_r16_lr1e4"
echo "  Config 5: Exit code $STATUS5 - models/config5_r1_alpha2_lr2e4_maxsteps60"
echo "  Config 6: Exit code $STATUS6 - models/config6_r4_alpha8_lr2e4"
echo "  Config 7: Exit code $STATUS7 - models/config7_r64_alpha128_lr2e4"
echo "  Config 8: Exit code $STATUS8 - models/config8_r1_alpha4_lr3e4"
echo ""
echo "Logs saved in logs/ directory"
echo "Models saved in models/ directory"
echo ""

