#!/usr/bin/env python3
"""
Single-GPU Training Script with Configurable Hyperparameters
Includes Weights & Biases logging
"""

import os
import argparse
import wandb

# Parse command line arguments early so we can set environment variables before
# importing any GPU-dependent libraries like torch/transformers/unsloth.
parser = argparse.ArgumentParser(description='Train with specific hyperparameters')
parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use')
parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size per device')
parser.add_argument('--grad_accum', type=int, default=4, help='Gradient accumulation steps')
parser.add_argument('--max_seq_length', type=int, default=1024, help='Max sequence length')
parser.add_argument('--output_name', type=str, required=True, help='Output directory name')
parser.add_argument('--train_samples', type=int, default=10000, help='Number of training samples')
parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (ignored if max_steps is provided)')
parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of training steps (takes precedence over epochs)')
parser.add_argument('--lora_alpha', type=int, default=None, help='LoRA alpha (defaults to 2 * lora_r if not specified)')
parser.add_argument('--wandb_project', type=str, default='math-verification', help='W&B project name')

args = parser.parse_args()

# Ensure each process only sees its assigned GPU BEFORE importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Determine lora_alpha
lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r * 2

# Initialize Weights & Biases
wandb_config = {
    "gpu_id": args.gpu_id,
    "lora_r": args.lora_r,
    "lora_alpha": lora_alpha,
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size,
    "grad_accum": args.grad_accum,
    "max_seq_length": args.max_seq_length,
    "train_samples": args.train_samples,
}
if args.max_steps is not None:
    wandb_config["max_steps"] = args.max_steps
else:
    wandb_config["epochs"] = args.epochs if args.epochs is not None else 3

wandb.init(
    entity="projectsolomon",  # Your W&B entity
    project=args.wandb_project,
    name=args.output_name,
    config=wandb_config
)

print(f"\n{'='*80}")
print(f"Training Configuration: {args.output_name}")
print(f"GPU: {args.gpu_id} | LoRA r: {args.lora_r} | LoRA alpha: {lora_alpha} | LR: {args.learning_rate}")
print(f"Batch size: {args.batch_size} | Grad accum: {args.grad_accum}")
print(f"Max seq length: {args.max_seq_length} | Samples: {args.train_samples}")
if args.max_steps is not None:
    print(f"Training duration: {args.max_steps} steps")
else:
    print(f"Training duration: {args.epochs if args.epochs is not None else 3} epochs")
print(f"{'='*80}\n")

# ============================================================================
# Model Configuration
# ============================================================================
dtype = None  # Auto-detect
load_in_4bit = True  # Use 4-bit quantization

print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=args.max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ============================================================================
# Dataset Preparation
# ============================================================================
print("Loading dataset...")
full_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")

# Shuffle and create splits
shuffled_dataset = full_dataset.shuffle(seed=42)
VAL_SIZE = 500

train_dataset = shuffled_dataset.select(range(args.train_samples))
validation_dataset = shuffled_dataset.select(range(args.train_samples, args.train_samples + VAL_SIZE))

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")

# ============================================================================
# Prompt Template and Formatting
# ============================================================================
training_prompt = """You are a great mathematician and you are tasked with finding if a solution to a given maths question is correct or not. Your response should be 'True' if the solution is correct, otherwise 'False'. Below is the Question and Solution.
Question:
{}
Solution:
{}
Output:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    questions = examples["question"]
    solutions = examples["solution"]
    outputs = examples["is_correct"]
    texts = []
    for question, solution, output in zip(questions, solutions, outputs):
        text = training_prompt.format(question, str(solution), str(output)) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

print("Formatting training dataset...")
formatted_train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

# ============================================================================
# LoRA Configuration
# ============================================================================
print("Setting up LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_r,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_alpha,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ============================================================================
# Training Configuration
# ============================================================================
print("Configuring trainer...")
# Build training args
training_args_dict = {
    # Batch Settings
    "per_device_train_batch_size": args.batch_size,
    "gradient_accumulation_steps": args.grad_accum,
    
    # Optimizer Settings
    "learning_rate": args.learning_rate,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "warmup_steps": 5,
    "lr_scheduler_type": "linear",
}
# Training Duration - max_steps takes precedence
if args.max_steps is not None:
    training_args_dict["max_steps"] = args.max_steps
else:
    training_args_dict["num_train_epochs"] = args.epochs if args.epochs is not None else 3

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train_dataset,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    args=TrainingArguments(
        **training_args_dict,
        
        # Precision
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        
        # Logging and Checkpoints
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,  # Only keep last checkpoint to save space
        
        # Output
        output_dir=f"outputs/{args.output_name}",
        report_to="wandb",  # Enable W&B logging
        run_name=args.output_name,
        seed=42,
    ),
)

# ============================================================================
# Training
# ============================================================================
print("\n" + "="*80)
print(f"Starting training: {args.output_name}")
print("="*80 + "\n")

trainer.train()

print("\n" + "="*80)
print(f"Training completed: {args.output_name}")
print("="*80 + "\n")

# ============================================================================
# Save Model
# ============================================================================
print("Saving model...")
model.save_pretrained(f"models/{args.output_name}")
tokenizer.save_pretrained(f"models/{args.output_name}")

print(f"Done! Model saved to ./models/{args.output_name}")

# Finish W&B run
wandb.finish()



