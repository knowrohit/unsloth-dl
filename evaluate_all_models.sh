#!/bin/bash
# Evaluate 4 different model configurations in parallel, one on each GPU

echo "=================================="
echo "Parallel Multi-Model Evaluation"
echo "Evaluating 4 different models simultaneously"
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
echo "Starting 4 parallel evaluation jobs..."
echo "=================================="
echo ""

# Configuration 1: High rank, standard learning rate
echo "Evaluating Config 1 (GPU 0): models/config1_r32_lr2e4"
CUDA_VISIBLE_DEVICES=0 python3 evaluate_model.py \
    --model_path models/config1_r32_lr2e4 \
    --gpu_id 0 \
    --max_seq_length 1024 \
    --eval_samples 500 \
    --generate_submission \
    > logs/eval_config1.log 2>&1 &

PID1=$!
echo "  Started Config 1 evaluation (PID: $PID1)"

# Configuration 2: Medium rank, higher learning rate
echo "Evaluating Config 2 (GPU 1): models/config2_r16_lr3e4"
CUDA_VISIBLE_DEVICES=1 python3 evaluate_model.py \
    --model_path models/config2_r16_lr3e4 \
    --gpu_id 1 \
    --max_seq_length 1024 \
    --eval_samples 500 \
    --generate_submission \
    > logs/eval_config2.log 2>&1 &

PID2=$!
echo "  Started Config 2 evaluation (PID: $PID2)"

# Configuration 3: Lower rank, standard LR, shorter sequences
echo "Evaluating Config 3 (GPU 2): models/config3_r8_lr2e4_seq512"
CUDA_VISIBLE_DEVICES=2 python3 evaluate_model.py \
    --model_path models/config3_r8_lr2e4_seq512 \
    --gpu_id 2 \
    --max_seq_length 512 \
    --eval_samples 500 \
    --generate_submission \
    > logs/eval_config3.log 2>&1 &

PID3=$!
echo "  Started Config 3 evaluation (PID: $PID3)"

# Configuration 4: Medium rank, lower LR (more conservative)
echo "Evaluating Config 4 (GPU 3): models/config4_r16_lr1e4"
CUDA_VISIBLE_DEVICES=3 python3 evaluate_model.py \
    --model_path models/config4_r16_lr1e4 \
    --gpu_id 3 \
    --max_seq_length 1024 \
    --eval_samples 500 \
    --generate_submission \
    > logs/eval_config4.log 2>&1 &

PID4=$!
echo "  Started Config 4 evaluation (PID: $PID4)"

echo ""
echo "=================================="
echo "All 4 evaluation jobs launched!"
echo "=================================="
echo ""
echo "Process IDs:"
echo "  Config 1 (GPU 0): $PID1"
echo "  Config 2 (GPU 1): $PID2"
echo "  Config 3 (GPU 2): $PID3"
echo "  Config 4 (GPU 3): $PID4"
echo ""
echo "Monitor logs with:"
echo "  tail -f logs/eval_config1.log"
echo "  tail -f logs/eval_config2.log"
echo "  tail -f logs/eval_config3.log"
echo "  tail -f logs/eval_config4.log"
echo ""
echo "Monitor GPU usage with:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Waiting for all evaluation jobs to complete..."
echo ""

# Wait for all background jobs
wait $PID1
STATUS1=$?
echo "Config 1 evaluation completed (exit code: $STATUS1)"

wait $PID2
STATUS2=$?
echo "Config 2 evaluation completed (exit code: $STATUS2)"

wait $PID3
STATUS3=$?
echo "Config 3 evaluation completed (exit code: $STATUS3)"

wait $PID4
STATUS4=$?
echo "Config 4 evaluation completed (exit code: $STATUS4)"

echo ""
echo "=================================="
echo "All evaluation jobs completed!"
echo "=================================="
echo ""
echo "Results:"
echo "  Config 1: Exit code $STATUS1"
echo "    - Validation results: models/config1_r32_lr2e4/validation_results.json"
echo "    - Submission file:    models/config1_r32_lr2e4/submission.csv"
echo ""
echo "  Config 2: Exit code $STATUS2"
echo "    - Validation results: models/config2_r16_lr3e4/validation_results.json"
echo "    - Submission file:    models/config2_r16_lr3e4/submission.csv"
echo ""
echo "  Config 3: Exit code $STATUS3"
echo "    - Validation results: models/config3_r8_lr2e4_seq512/validation_results.json"
echo "    - Submission file:    models/config3_r8_lr2e4_seq512/submission.csv"
echo ""
echo "  Config 4: Exit code $STATUS4"
echo "    - Validation results: models/config4_r16_lr1e4/validation_results.json"
echo "    - Submission file:    models/config4_r16_lr1e4/submission.csv"
echo ""
echo "Evaluation logs saved in logs/ directory"
echo ""
echo "To compare results, run:"
echo "  cat models/config*/validation_results.json | grep -E '(model_path|accuracy|f1_score)'"
echo ""


