#!/usr/bin/env python3
"""
Single-GPU Training Script for Math Answer Verification
Matches the Colab setup with 10,000 training samples
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================================
# Model Configuration
# ============================================================================
max_seq_length = 1024  # Match Colab setup
dtype = None  # Auto-detect
load_in_4bit = True  # Use 4-bit quantization

print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ============================================================================
# Dataset Preparation
# ============================================================================
print("Loading dataset...")
full_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")

# Shuffle and create splits (10,000 training + 500 validation)
shuffled_dataset = full_dataset.shuffle(seed=42)
TRAIN_SIZE = 10000
VAL_SIZE = 500

train_dataset = shuffled_dataset.select(range(TRAIN_SIZE))
validation_dataset = shuffled_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))

print(f"Total dataset size: {len(full_dataset)}")
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
    r=16,  # Increased from Colab's r=1 for better capacity
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,  # 2 * r
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ============================================================================
# Training Configuration (Single GPU)
# ============================================================================
print("Configuring trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        # Batch Settings (matching Colab effective batch size)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        
        # Training Duration
        num_train_epochs=3,
        
        # Optimizer Settings
        learning_rate=2e-4,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=5,
        lr_scheduler_type="linear",
        
        # Precision
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        
        # Logging and Checkpoints
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        
        # Output
        output_dir="outputs_single_gpu",
        report_to="none",
        seed=42,
    ),
)

# ============================================================================
# Training
# ============================================================================
print("\n" + "="*80)
print("Starting training on single GPU...")
print("="*80 + "\n")

trainer.train()

print("\n" + "="*80)
print("Training completed!")
print("="*80 + "\n")

# ============================================================================
# Save Model
# ============================================================================
print("Saving model...")
model.save_pretrained("final_model_single_gpu")
tokenizer.save_pretrained("final_model_single_gpu")

print("Done! Model saved to ./final_model_single_gpu")

