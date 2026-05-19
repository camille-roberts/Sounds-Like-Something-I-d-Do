#!/usr/bin/env python3
"""
Camille Comedy Assistant - Fine-tuning Script (Updated for Improved Dataset)

Based on proven successful configuration with Mistral-7B.
Uses improved dataset with long-form continuations and Q&A pairs.

Usage:
    python train_improved.py --data-file training_data_improved.jsonl --era era_one --output-dir ./adapters/era_one
    python train_improved.py --data-file training_data_improved.jsonl --era all --output-dir ./adapters/all_eras
"""

import argparse
import json
import os
import sys
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from sklearn.model_selection import train_test_split


# System prompt for training
SYSTEM_PROMPT = """You are Camille's comedy writing assistant. Your role is to continue jokes and respond in Camille's unique voice and style. 

Guidelines:
- Match Camille's observational comedy style, phrasing, and expressions exactly
- Draw from her characteristic humor patterns and comedic timing
- Keep responses authentic to her voice - be indistinguishable from her writing
- Use her narrative style and typical expressions
- Minimize generic or novel content - stay true to her established comedic persona"""


def load_and_filter_data(data_file: str, era: str) -> list[dict]:
    """Load JSONL data and filter by era."""
    
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line: {e}")
                continue
    
    print(f"Loaded {len(data)} total examples from {data_file}")
    
    # Filter by era if not "all"
    if era.lower() != "all":
        filtered_data = [
            item for item in data
            if item.get("meta", {}).get("era", "").lower() == era.lower()
        ]
        print(f"Filtered to {len(filtered_data)} examples for era '{era}'")
        return filtered_data
    
    print(f"Using all {len(data)} examples (all eras)")
    return data


def apply_tag_weighting(data: list[dict]) -> list[dict]:
    """
    Apply tag-based weighting by duplicating entries.
    transcripts: 4x, jokes: 3x, memoir: 1x, thoughts: 1x
    """
    
    weighted_data = []
    weight_map = {
        'transcripts': 4.0,
        'jokes': 3.0,
        'memoir': 1.0,
        'poetry': 1.0,
        'thoughts': 1.0
    }
    
    for item in data:
        tag = item.get("meta", {}).get("tag", "transcripts")
        weight = weight_map.get(tag, 1.0)
        
        # Add the item multiple times based on weight
        count = int(weight)
        for _ in range(count):
            weighted_data.append(item)
    
    print(f"Applied tag weighting: {len(data)} → {len(weighted_data)} effective examples")
    return weighted_data


def format_conversation(item: dict, tokenizer, max_length: int = 2048) -> dict:
    """Format a single conversation for training using chat template."""
    
    messages = item.get("messages", [])
    
    # Build the conversation with system prompt
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["user", "assistant"] and content:
            formatted_messages.append({"role": role, "content": content})
    
    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except Exception:
        # Fallback for tokenizers without chat template
        text = ""
        for msg in formatted_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                text += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}</s>\n"
    
    return {"text": text}


def tokenize_function(examples: dict, tokenizer, max_length: int = 2048) -> dict:
    """Tokenize examples for training."""
    
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def prepare_datasets(
    data: list[dict],
    tokenizer,
    validation_split: float = 0.1,
    max_length: int = 2048,
    seed: int = 42,
    apply_weighting: bool = True
) -> tuple[Dataset, Dataset]:
    """Prepare training and validation datasets."""
    
    # Split data first
    train_data, val_data = train_test_split(
        data, 
        test_size=validation_split, 
        random_state=seed
    )
    
    print(f"Initial split: {len(train_data)} train, {len(val_data)} validation")
    
    # Apply weighting to training data only
    if apply_weighting:
        train_data = apply_tag_weighting(train_data)
    
    print(f"Final training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Format conversations
    train_formatted = [format_conversation(item, tokenizer, max_length) for item in train_data]
    val_formatted = [format_conversation(item, tokenizer, max_length) for item in val_data]
    
    # Create datasets
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        remove_columns=["text"],
        desc="Tokenizing training data"
    )
    
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        remove_columns=["text"],
        desc="Tokenizing validation data"
    )
    
    return train_dataset, val_dataset


def setup_model_and_tokenizer(
    model_path: str = "mistralai/Mistral-7B-Instruct-v0.3",
    use_quantization: bool = False
) -> tuple:
    """Load and configure model and tokenizer."""
    
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Quantization config (optional - useful for smaller GPUs)
    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model
    # Try flash_attention_2, fall back to sdpa, then default
    attn_implementation = None
    if torch.cuda.is_available():
        try:
            import flash_attn
            attn_implementation = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            attn_implementation = "sdpa"  # PyTorch native scaled dot product attention
            print("Flash Attention not installed, using SDPA (still fast)")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
    )
    
    # Prepare for training if using quantization
    if use_quantization:
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer


def setup_lora(model, lora_rank: int = 64, lora_alpha: int = 128) -> object:
    """Configure and apply LoRA adapters."""
    
    # LoRA configuration - proven settings from successful training
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    
    return model


def train(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 3,
    max_length: int = 2048,
):
    """Train the model."""
    
    # Calculate effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"\nEffective batch size: {effective_batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Training arguments - proven configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=max_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.001
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )
    
    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    trainer.train()
    
    # Save final adapter
    print(f"\nSaving adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training config for reference
    config = {
        "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "dataset": "training_data_improved.jsonl",
        "lora_rank": model.peft_config["default"].r,
        "lora_alpha": model.peft_config["default"].lora_alpha,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Adapter saved to: {output_dir}")
    print("=" * 50)
    
    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral-7B on improved Camille comedy data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on era_one data
  python train_improved.py --data-file training_data_improved.jsonl --era era_one --output-dir ./adapters/era_one

  # Train on era_two data  
  python train_improved.py --data-file training_data_improved.jsonl --era era_two --output-dir ./adapters/era_two

  # Train on era_three data
  python train_improved.py --data-file training_data_improved.jsonl --era era_three --output-dir ./adapters/era_three

  # Train on all data combined
  python train_improved.py --data-file training_data_improved.jsonl --era all --output-dir ./adapters/all_eras

  # Custom parameters
  python train_improved.py --data-file training_data_improved.jsonl --era all --epochs 15 --batch-size 2
        """
    )
    
    # Data arguments
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to JSONL file containing training data"
    )
    
    parser.add_argument(
        "--era",
        type=str,
        required=True,
        choices=["era_one", "era_two", "era_three", "all"],
        help="Which era to train on (era_one, era_two, era_three, or all)"
    )
    
    # Model arguments
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Path to base model or HuggingFace model name (default: mistralai/Mistral-7B-Instruct-v0.3)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the trained adapter"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs (default: 10)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device (default: 4)"
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1)"
    )
    
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience in eval steps (default: 3)"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA rank (default: 64)"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=128,
        help="LoRA alpha (default: 128)"
    )
    
    # Other arguments
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Use 4-bit quantization (saves memory but slower)"
    )
    
    parser.add_argument(
        "--no-weighting",
        action="store_true",
        help="Disable tag-based weighting (transcripts 4x, jokes 3x, etc.)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Validate inputs
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)
    
    # Load and filter data
    print("\n" + "=" * 50)
    print(f"Training Camille Comedy Assistant - Era: {args.era}")
    print("=" * 50)
    
    data = load_and_filter_data(args.data_file, args.era)
    
    if len(data) == 0:
        print(f"Error: No data found for era '{args.era}'")
        sys.exit(1)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.base_model,
        use_quantization=args.use_quantization
    )
    
    # Setup LoRA
    model = setup_lora(model, args.lora_rank, args.lora_alpha)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(
        data,
        tokenizer,
        validation_split=args.validation_split,
        max_length=args.max_length,
        seed=args.seed,
        apply_weighting=not args.no_weighting
    )
    
    # Train
    train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
