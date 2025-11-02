#!/usr/bin/env python3
"""
Evaluation Script for Trained Models
Evaluates on validation set and generates submission file
"""

import torch
import argparse
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import load_dataset
import json

# Try importing sklearn, if not available use manual calculation
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not found, will calculate metrics manually")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate a trained model')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
parser.add_argument('--max_seq_length', type=int, default=1024, help='Max sequence length')
parser.add_argument('--eval_samples', type=int, default=500, help='Number of validation samples')
parser.add_argument('--generate_submission', action='store_true', help='Generate submission file for test set')

args = parser.parse_args()

# Set GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

print(f"\n{'='*80}")
print(f"Evaluating Model: {args.model_path}")
print(f"GPU: {args.gpu_id}")
print(f"{'='*80}\n")

# ============================================================================
# Load Model
# ============================================================================
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_path,
    max_seq_length=args.max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# Prepare model for inference
FastLanguageModel.for_inference(model)
print("Model loaded and ready for inference!\n")

# ============================================================================
# Inference Prompt Template
# ============================================================================
inference_prompt = """You are a great mathematician and you are tasked with finding if a solution to a given maths question is correct or not. Your response should be 'True' if the solution is correct, otherwise 'False'. Below is the Question and Solution.
Question:
{}
Solution:
{}
Output:
"""

# ============================================================================
# Helper Function to Parse Model Output
# ============================================================================
def parse_output(response_text):
    """Parse 'True' or 'False' from the model's raw output"""
    # Find the text after "Output:"
    output_part = response_text.split("Output:\n")[-1]
    # Check if "True" is in that part, case-insensitively
    if 'true' in output_part.lower():
        return True
    return False

# ============================================================================
# Evaluate on Validation Set
# ============================================================================
print("Loading validation dataset...")
full_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")
shuffled_dataset = full_dataset.shuffle(seed=42)

# Use same split as training (skip first 10000 for training, take next 500 for validation)
TRAIN_SIZE = 10000
validation_dataset = shuffled_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + args.eval_samples))

print(f"Validation samples: {len(validation_dataset)}\n")

print("Running inference on validation set...")
predictions = []
ground_truth = []

for example in tqdm(validation_dataset):
    question = example["question"]
    solution = example["solution"]
    correct_answer = example["is_correct"]
    
    # Format the prompt
    prompt = inference_prompt.format(question, str(solution))
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate the prediction
    outputs = model.generate(**inputs, max_new_tokens=8, use_cache=True)
    response_text = tokenizer.batch_decode(outputs)[0]
    
    # Parse the prediction
    prediction = parse_output(response_text)
    predictions.append(prediction)
    ground_truth.append(correct_answer)

# ============================================================================
# Calculate Metrics
# ============================================================================
print("\n" + "="*80)
print("VALIDATION RESULTS")
print("="*80)

if HAS_SKLEARN:
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='binary', zero_division=0
    )
    cm = confusion_matrix(ground_truth, predictions)
else:
    # Manual calculation
    correct = sum([1 for gt, pred in zip(ground_truth, predictions) if gt == pred])
    accuracy = correct / len(ground_truth)
    
    # Calculate TP, FP, TN, FN
    tp = sum([1 for gt, pred in zip(ground_truth, predictions) if gt == True and pred == True])
    fp = sum([1 for gt, pred in zip(ground_truth, predictions) if gt == False and pred == True])
    tn = sum([1 for gt, pred in zip(ground_truth, predictions) if gt == False and pred == False])
    fn = sum([1 for gt, pred in zip(ground_truth, predictions) if gt == True and pred == False])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    cm = [[tn, fp], [fn, tp]]

print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print("                Predicted")
print("                False   True")
print(f"Actual False    {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"Actual True     {cm[1][0]:5d}  {cm[1][1]:5d}")

# Save results
results = {
    "model_path": args.model_path,
    "validation_samples": len(validation_dataset),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist() if HAS_SKLEARN else cm
}

results_file = f"{args.model_path}/validation_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {results_file}")

# ============================================================================
# Generate Submission File (Optional)
# ============================================================================
if args.generate_submission:
    print("\n" + "="*80)
    print("GENERATING SUBMISSION FILE")
    print("="*80 + "\n")
    
    # Load test dataset
    test_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")
    print(f"Test samples: {len(test_dataset)}")
    
    print("Running inference on test set...")
    test_predictions = []
    
    for example in tqdm(test_dataset):
        question = example["question"]
        solution = example["solution"]
        
        # Format the prompt
        prompt = inference_prompt.format(question, str(solution))
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # Generate the prediction
        outputs = model.generate(**inputs, max_new_tokens=8, use_cache=True)
        response_text = tokenizer.batch_decode(outputs)[0]
        
        # Parse the prediction
        prediction = parse_output(response_text)
        test_predictions.append(prediction)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'ID': range(len(test_predictions)),
        'is_correct': test_predictions
    })
    
    # Save submission file
    submission_file = f"{args.model_path}/submission.csv"
    submission.to_csv(submission_file, index=False)
    
    print(f"\nSubmission file created: {submission_file}")
    print(f"Total predictions: {len(test_predictions)}")
    print(f"Predicted True:  {sum(test_predictions)} ({sum(test_predictions)/len(test_predictions)*100:.2f}%)")
    print(f"Predicted False: {len(test_predictions)-sum(test_predictions)} ({(len(test_predictions)-sum(test_predictions))/len(test_predictions)*100:.2f}%)")

print("\n" + "="*80)
print("EVALUATION COMPLETED!")
print("="*80 + "\n")

