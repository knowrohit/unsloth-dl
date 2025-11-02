#!/bin/bash
# Evaluate configurations 5-8 in parallel, one on each GPU

echo "=================================="
echo "Parallel Multi-Model Evaluation"
echo "Evaluating 4 different models (Configs 5-8) simultaneously"
echo "=================================="

# Create output directories if needed
mkdir -p models
mkdir -p logs

# Check GPU availability
echo ""
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "=================================="
echo "Starting 4 parallel evaluation jobs (Configs 5-8)..."
echo "=================================="
echo ""

# Configuration 5: Very low rank (r=1), max_steps=60, lora_alpha=2
echo "Evaluating Config 5 (GPU 0): models/config5_r1_alpha2_lr2e4_maxsteps60"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_model.py \
    --model_path models/config5_r1_alpha2_lr2e4_maxsteps60 \
    --gpu_id 0 \
    --max_seq_length 1024 \
    --eval_samples 500 \
    --generate_submission \
    > logs/eval_config5.log 2>&1 &

PID5=$!
echo "  Started Config 5 evaluation (PID: $PID5)"

# Configuration 6: Low rank r=4, alpha=8, epochs
echo "Evaluating Config 6 (GPU 1): models/config6_r4_alpha8_lr2e4"
CUDA_VISIBLE_DEVICES=1 python3 evaluate_model.py \
    --model_path models/config6_r4_alpha8_lr2e4 \
    --gpu_id 1 \
    --max_seq_length 1024 \
    --eval_samples 500 \
    --generate_submission \
    > logs/eval_config6.log 2>&1 &

PID6=$!
echo "  Started Config 6 evaluation (PID: $PID6)"

# Configuration 7: Very high rank r=64, alpha=128, epochs
echo "Evaluating Config 7 (GPU 2): models/config7_r64_alpha128_lr2e4"
CUDA_VISIBLE_DEVICES=2 python3 evaluate_model.py \
    --model_path models/config7_r64_alpha128_lr2e4 \
    --gpu_id 2 \
    --max_seq_length 1024 \
    --eval_samples 500 \
    --generate_submission \
    > logs/eval_config7.log 2>&1 &

PID7=$!
echo "  Started Config 7 evaluation (PID: $PID7)"

# Configuration 8: Low rank r=1, alpha=4, higher LR, epochs
echo "Evaluating Config 8 (GPU 3): models/config8_r1_alpha4_lr3e4"
CUDA_VISIBLE_DEVICES=3 python3 evaluate_model.py \
    --model_path models/config8_r1_alpha4_lr3e4 \
    --gpu_id 3 \
    --max_seq_length 1024 \
    --eval_samples 500 \
    --generate_submission \
    > logs/eval_config8.log 2>&1 &

PID8=$!
echo "  Started Config 8 evaluation (PID: $PID8)"

echo ""
echo "=================================="
echo "All 4 evaluation jobs launched!"
echo "=================================="
echo ""
echo "Process IDs:"
echo "  Config 5 (GPU 0): $PID5"
echo "  Config 6 (GPU 1): $PID6"
echo "  Config 7 (GPU 2): $PID7"
echo "  Config 8 (GPU 3): $PID8"
echo ""
echo "Monitor logs with:"
echo "  tail -f logs/eval_config5.log"
echo "  tail -f logs/eval_config6.log"
echo "  tail -f logs/eval_config7.log"
echo "  tail -f logs/eval_config8.log"
echo ""
echo "Monitor GPU usage with:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Waiting for all evaluation jobs to complete..."
echo ""

# Wait for all background jobs
wait $PID5
STATUS5=$?
echo "Config 5 evaluation completed (exit code: $STATUS5)"

wait $PID6
STATUS6=$?
echo "Config 6 evaluation completed (exit code: $STATUS6)"

wait $PID7
STATUS7=$?
echo "Config 7 evaluation completed (exit code: $STATUS7)"

wait $PID8
STATUS8=$?
echo "Config 8 evaluation completed (exit code: $STATUS8)"

echo ""
echo "=================================="
echo "All evaluation jobs completed!"
echo "=================================="
echo ""
echo "Results:"
echo "  Config 5: Exit code $STATUS5"
echo "    - Validation results: models/config5_r1_alpha2_lr2e4_maxsteps60/validation_results.json"
echo "    - Submission file:    models/config5_r1_alpha2_lr2e4_maxsteps60/submission.csv"
echo ""
echo "  Config 6: Exit code $STATUS6"
echo "    - Validation results: models/config6_r4_alpha8_lr2e4/validation_results.json"
echo "    - Submission file:    models/config6_r4_alpha8_lr2e4/submission.csv"
echo ""
echo "  Config 7: Exit code $STATUS7"
echo "    - Validation results: models/config7_r64_alpha128_lr2e4/validation_results.json"
echo "    - Submission file:    models/config7_r64_alpha128_lr2e4/submission.csv"
echo ""
echo "  Config 8: Exit code $STATUS8"
echo "    - Validation results: models/config8_r1_alpha4_lr3e4/validation_results.json"
echo "    - Submission file:    models/config8_r1_alpha4_lr3e4/submission.csv"
echo ""
echo "Evaluation logs saved in logs/ directory"
echo ""
echo "To compare results, run:"
echo "  cat models/config{5,6,7,8}*/validation_results.json | grep -E '(model_path|accuracy|f1_score)'"
echo ""

