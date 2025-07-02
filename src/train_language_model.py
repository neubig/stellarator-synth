#!/usr/bin/env python3
"""
Train a language model on stellarator text data.
This script fine-tunes a pre-trained language model on the generated text dataset.
"""

import os
import argparse
import logging
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("language_model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        Trainer, 
        TrainingArguments,
        DataCollatorForLanguageModeling
    )
    from datasets import load_dataset
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please install required packages: pip install torch transformers datasets")
    raise


class StellaratorTextDataset(Dataset):
    """Dataset for stellarator text data."""
    
    def __init__(self, text_file, tokenizer, max_length=1024):
        """
        Initialize the dataset.
        
        Args:
            text_file: Path to text file
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading text from {text_file}")
        with open(text_file, 'r') as f:
            self.text = f.read()
        
        # Split into chunks of max_length tokens
        self.chunks = self._create_chunks()
        logger.info(f"Created {len(self.chunks)} text chunks")
    
    def _create_chunks(self):
        """Split the text into chunks of max_length tokens."""
        tokens = self.tokenizer.encode(self.text)
        chunks = []
        
        for i in range(0, len(tokens), self.max_length):
            chunk = tokens[i:i + self.max_length]
            if len(chunk) > 100:  # Only keep chunks with a reasonable length
                chunks.append(chunk)
        
        return chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {"input_ids": chunk, "attention_mask": [1] * len(chunk)}


def train_language_model(
    text_file,
    model_name="gpt2",
    output_dir="models/language_model",
    batch_size=4,
    learning_rate=5e-5,
    num_epochs=3,
    max_length=1024,
    save_steps=500,
    warmup_steps=500,
    use_hf_datasets=True
):
    """
    Train a language model on stellarator text data.
    
    Args:
        text_file: Path to text file
        model_name: Pre-trained model name
        output_dir: Directory to save the model
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        max_length: Maximum sequence length
        save_steps: Save checkpoint every N steps
        warmup_steps: Number of warmup steps
        use_hf_datasets: Whether to use HuggingFace datasets library
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    if use_hf_datasets:
        # Create a temporary file with the dataset format HF expects
        temp_dir = os.path.join(output_dir, "temp_dataset")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy the text file to the temp directory
        import shutil
        temp_file = os.path.join(temp_dir, "text.txt")
        shutil.copy(text_file, temp_file)
        
        # Load dataset
        logger.info("Loading dataset using HuggingFace datasets")
        dataset = load_dataset("text", data_files={"train": temp_file})
        
        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're doing causal language modeling
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=save_steps,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
        )
        
        # Train the model
        logger.info("Starting training")
        trainer.train()
        
        # Save the model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
    else:
        # Use custom dataset
        logger.info("Using custom dataset implementation")
        dataset = StellaratorTextDataset(text_file, tokenizer, max_length)
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Training setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Training loop
        logger.info("Starting training")
        model.train()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            total_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # For causal language modeling
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
                
                # Save checkpoint
                if (batch_idx + 1) % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}-{batch_idx+1}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        # Save the final model
        logger.info(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # Save training metadata
    metadata = {
        "model_name": model_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "max_length": max_length,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Training completed successfully")


def train_prompt_completion_model(
    prompt_file,
    completion_file,
    model_name="gpt2",
    output_dir="models/language_model",
    batch_size=4,
    learning_rate=5e-5,
    num_epochs=3,
    max_length=1024,
    save_steps=500,
    warmup_steps=500
):
    """
    Train a language model on prompt-completion pairs.
    
    Args:
        prompt_file: Path to prompts file
        completion_file: Path to completions file
        model_name: Pre-trained model name
        output_dir: Directory to save the model
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        max_length: Maximum sequence length
        save_steps: Save checkpoint every N steps
        warmup_steps: Number of warmup steps
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts and completions
    logger.info(f"Loading prompts from {prompt_file}")
    with open(prompt_file, 'r') as f:
        prompts = f.readlines()
    
    logger.info(f"Loading completions from {completion_file}")
    with open(completion_file, 'r') as f:
        completions = f.readlines()
    
    if len(prompts) != len(completions):
        logger.error(f"Number of prompts ({len(prompts)}) doesn't match number of completions ({len(completions)})")
        return
    
    # Create a temporary file with the combined data
    temp_dir = os.path.join(output_dir, "temp_dataset")
    os.makedirs(temp_dir, exist_ok=True)
    
    combined_file = os.path.join(temp_dir, "combined.txt")
    with open(combined_file, 'w') as f:
        for prompt, completion in zip(prompts, completions):
            f.write(f"{prompt.strip()}\n{completion.strip()}\n\n")
    
    # Train the model using the combined file
    train_language_model(
        text_file=combined_file,
        model_name=model_name,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_length=max_length,
        save_steps=save_steps,
        warmup_steps=warmup_steps
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train language model on stellarator text data")
    
    # Common arguments
    parser.add_argument("--model_name", type=str, default="gpt2", help="Pre-trained model name")
    parser.add_argument("--output_dir", type=str, default="models/language_model", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["text", "prompt"], default="text",
                       help="Training mode: 'text' for continuous text, 'prompt' for prompt-completion pairs")
    
    # Mode-specific arguments
    parser.add_argument("--text_file", type=str, help="Path to text file (for 'text' mode)")
    parser.add_argument("--prompt_file", type=str, help="Path to prompts file (for 'prompt' mode)")
    parser.add_argument("--completion_file", type=str, help="Path to completions file (for 'prompt' mode)")
    
    args = parser.parse_args()
    
    if args.mode == "text":
        if not args.text_file:
            parser.error("--text_file is required for 'text' mode")
        
        train_language_model(
            text_file=args.text_file,
            model_name=args.model_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            save_steps=args.save_steps,
            warmup_steps=args.warmup_steps
        )
    else:  # prompt mode
        if not args.prompt_file or not args.completion_file:
            parser.error("--prompt_file and --completion_file are required for 'prompt' mode")
        
        train_prompt_completion_model(
            prompt_file=args.prompt_file,
            completion_file=args.completion_file,
            model_name=args.model_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            save_steps=args.save_steps,
            warmup_steps=args.warmup_steps
        )