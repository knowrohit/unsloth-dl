#!/usr/bin/env python3
"""
Multi-GPU Training Script for Math Answer Verification
Optimized for Unsloth distributed training with 4x RTX 5060 Ti (12GB VRAM each)
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch.distributed as dist

# ============================================================================
# Memory Optimization Settings
# ============================================================================
# Enable expandable segments to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Disable Unsloth's fused cross-entropy loss (can cause issues with DDP)
os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"

# Set up distributed environment if available
device_map = None
local_rank = 0
try:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        # Set the device FIRST before any CUDA operations
        torch.cuda.set_device(local_rank)
        # Use local_rank directly to ensure correct device
        device_map = {"": local_rank}  # Load entire model on this GPU
        # Small stagger to prevent all processes loading simultaneously
        import time
        time.sleep(local_rank * 0.5)  # 0s, 0.5s, 1s, 1.5s delays
        print(f"[Rank {local_rank}] Process assigned to GPU {local_rank}, device_map={device_map}")
    else:
        # Single GPU mode
        if torch.cuda.is_available():
            device_map = {"": 0}
except Exception as e:
    print(f"Distributed setup note: {e}")
    # Fallback: use default device
    if torch.cuda.is_available():
        device_map = {"": 0}

# Clear cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# Model Configuration
# ============================================================================
max_seq_length = 512  # Reduced to 512 to fit in 12GB VRAM with 4 GPUs
dtype = None  # Auto-detect
load_in_4bit = True

print("Loading model and tokenizer...")
print(f"Loading on device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
# Each process will load on its assigned GPU using device_map
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map=device_map,  # Explicitly set device for distributed training
)

# Clear cache after model loading
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# Dataset Preparation
# ============================================================================
print("Loading dataset...")
full_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")

# Shuffle and create fixed-size splits (reduced for memory efficiency)
shuffled_dataset = full_dataset.shuffle(seed=42)
TRAIN_SIZE = 10000  # Fixed: 10,000 training samples
VAL_SIZE = 500      # Fixed: 500 validation samples

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
    r=16,  # Increased rank for better capacity
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,  # 2 * r
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ============================================================================
# Training Configuration (Multi-GPU Optimized)
# ============================================================================
print("Configuring trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        # Multi-GPU Settings
        ddp_find_unused_parameters=False,  # CRITICAL for DDP
        ddp_backend="nccl",  # Use NCCL for multi-GPU
        
        # Batch Settings (reduced for 12GB VRAM GPUs)
        per_device_train_batch_size=1,  # Reduced from 2 to 1 to save memory
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size (8 instead of 4)
        
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
        
        # Logging and Checkpoints (disabled during training to save memory)
        logging_steps=10,
        save_strategy="no",  # Disable checkpointing during training to save memory
        # save_steps=500,  # Commented out - no checkpoints during training
        # save_total_limit=3,  # Commented out - no checkpoints during training
        
        # Output
        output_dir="outputs",
        report_to="none",
        seed=42,
        
        # Performance
        dataloader_num_workers=2,  # Reduced to save memory
        group_by_length=False,  # Can enable for variable length optimization
        
        # Memory Optimization
        max_grad_norm=1.0,  # Gradient clipping to prevent memory spikes
        remove_unused_columns=True,  # Save memory
    ),
)

# ============================================================================
# Training
# ============================================================================
print("\n" + "="*80)
print("Starting distributed training...")
print("="*80 + "\n")

trainer.train()

# Wait for all processes to finish training
try:
    if dist.is_initialized():
        dist.barrier()
except:
    pass

print("\n" + "="*80)
print("Training completed!")
print("="*80 + "\n")

# ============================================================================
# Save Model (Only from rank 0 to avoid OOM)
# ============================================================================
# Check if we're in distributed mode and get rank
try:
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
except:
    is_distributed = False
    rank = 0

# Only save from the main process (rank 0) to avoid memory conflicts
if rank == 0:
    print("Saving model (only from rank 0)...")
    
    # Clear cache before saving
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Move model to CPU temporarily for saving to avoid GPU memory issues
    try:
        # Save to a temporary CPU location first if needed
        model.save_pretrained("final_model", safe_serialization=True)
        tokenizer.save_pretrained("final_model")
        print("Done! Model saved to ./final_model")
    except Exception as e:
        print(f"Error saving model: {e}")
        print("Attempting CPU offload save...")
        # Alternative: Try saving with device_map to CPU
        model.save_pretrained("final_model", safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained("final_model")
        print("Done! Model saved to ./final_model")
else:
    print(f"Rank {rank} skipping save (only rank 0 saves)")

# Final barrier to ensure save completes before processes exit
try:
    if is_distributed:
        dist.barrier()
except:
    pass

