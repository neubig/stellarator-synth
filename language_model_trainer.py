#!/usr/bin/env python3
"""
Language Model Trainer for Stellarator Physics

This module provides tools for training language models on stellarator simulation data
converted to text format.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# Optional: Transformers library for pre-trained models
try:
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        AutoTokenizer, AutoModelForCausalLM
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers library not available. Install with: pip install transformers")

# Optional: Datasets library
try:
    from datasets import Dataset as HFDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class StellaratorTextDataset(Dataset):
    """PyTorch Dataset for stellarator text data."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class StellaratorLanguageModel:
    """Language model for stellarator physics simulation."""
    
    def __init__(self, model_name: str = "gpt2", max_length: int = 512):
        """
        Initialize the language model.
        
        Args:
            model_name: Name of the pre-trained model to use
            max_length: Maximum sequence length
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required. Install with: pip install transformers")
        
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.training_stats = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_model(self, vocab_size: Optional[int] = None):
        """Setup the tokenizer and model."""
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens for stellarator data
        special_tokens = {
            'additional_special_tokens': [
                '<|startoftext|>', '<|endoftext|>',
                '<|format:structured|>', '<|format:natural|>', 
                '<|format:qa_pairs|>', '<|format:json_like|>', '<|format:code_like|>',
                'STELLARATOR_CONFIGURATION:', 'INPUT_PARAMETERS:', 'OUTPUT_METRICS:',
                'END_CONFIGURATION'
            ]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Resize token embeddings to accommodate new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.logger.info(f"Model setup complete. Vocabulary size: {len(self.tokenizer)}")
    
    def load_text_data(self, data_dir: str, format_type: str = 'combined') -> Tuple[List[str], List[str]]:
        """
        Load text data for training.
        
        Args:
            data_dir: Directory containing text data files
            format_type: Type of format to load ('structured', 'natural', 'qa_pairs', 'combined')
            
        Returns:
            Tuple of (train_texts, val_texts)
        """
        data_dir = Path(data_dir)
        
        # Load training data
        train_file = data_dir / f"train_{format_type}.txt"
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_text = f.read()
        
        # Load validation data
        val_file = data_dir / f"val_{format_type}.txt"
        if not val_file.exists():
            raise FileNotFoundError(f"Validation file not found: {val_file}")
        
        with open(val_file, 'r', encoding='utf-8') as f:
            val_text = f.read()
        
        # Split into individual examples
        train_texts = [text.strip() for text in train_text.split('\n\n') if text.strip()]
        val_texts = [text.strip() for text in val_text.split('\n\n') if text.strip()]
        
        self.logger.info(f"Loaded {len(train_texts)} training examples and {len(val_texts)} validation examples")
        
        return train_texts, val_texts
    
    def train_model(self, train_texts: List[str], val_texts: List[str], 
                   output_dir: str, training_args: Optional[Dict] = None):
        """
        Train the language model.
        
        Args:
            train_texts: List of training text examples
            val_texts: List of validation text examples
            output_dir: Directory to save the trained model
            training_args: Training arguments dictionary
        """
        if self.model is None or self.tokenizer is None:
            self.setup_model()
        
        # Default training arguments
        default_args = {
            'output_dir': output_dir,
            'overwrite_output_dir': True,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 4,
            'warmup_steps': 100,
            'logging_steps': 50,
            'save_steps': 500,
            'eval_steps': 500,
            'evaluation_strategy': 'steps',
            'save_strategy': 'steps',
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'fp16': torch.cuda.is_available(),
            'dataloader_num_workers': 2,
            'remove_unused_columns': False,
        }
        
        if training_args:
            default_args.update(training_args)
        
        # Create datasets
        train_dataset = StellaratorTextDataset(train_texts, self.tokenizer, self.max_length)
        val_dataset = StellaratorTextDataset(val_texts, self.tokenizer, self.max_length)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling
        )
        
        # Training arguments
        training_arguments = TrainingArguments(**default_args)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        self.logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training stats
        self.training_stats = {
            'train_loss': train_result.training_loss,
            'train_samples': len(train_texts),
            'val_samples': len(val_texts),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'training_args': default_args
        }
        
        stats_file = Path(output_dir) / "training_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        self.logger.info(f"Training completed. Model saved to: {output_dir}")
        self.logger.info(f"Final training loss: {train_result.training_loss:.4f}")
    
    def load_trained_model(self, model_dir: str):
        """Load a previously trained model."""
        model_dir = Path(model_dir)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        
        # Load training stats if available
        stats_file = model_dir / "training_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.training_stats = json.load(f)
        
        self.logger.info(f"Model loaded from: {model_dir}")
    
    def generate_prediction(self, prompt: str, max_new_tokens: int = 200, 
                          temperature: float = 0.7, do_sample: bool = True) -> str:
        """
        Generate a prediction given a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call setup_model() or load_trained_model() first.")
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def predict_stellarator_metrics(self, input_params: Dict[str, Any], 
                                  format_type: str = 'structured') -> Dict[str, Any]:
        """
        Predict stellarator metrics using the language model.
        
        Args:
            input_params: Dictionary of input parameters
            format_type: Format type for the prompt
            
        Returns:
            Dictionary of predicted metrics
        """
        from text_data_converter import StellaratorTextConverter
        
        # Create prompt
        converter = StellaratorTextConverter()
        prompt = converter.create_inference_prompt(input_params, format_type)
        
        # Generate prediction
        prediction = self.generate_prediction(prompt, max_new_tokens=300, temperature=0.1)
        
        # Parse prediction (this is format-specific)
        parsed_results = self._parse_prediction(prediction, format_type)
        
        return parsed_results
    
    def _parse_prediction(self, prediction: str, format_type: str) -> Dict[str, Any]:
        """Parse the model's prediction to extract numerical values."""
        
        results = {}
        
        if format_type == 'structured':
            # Parse structured format
            lines = prediction.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().replace(' ', '_')
                    value = value.strip()
                    
                    # Try to convert to float
                    try:
                        if value != 'N/A':
                            results[key] = float(value)
                    except ValueError:
                        results[key] = value
        
        elif format_type == 'natural':
            # Parse natural language using regex
            patterns = {
                'aspect_ratio_computed': r'computed aspect ratio of ([\d.]+)',
                'max_elongation': r'maximum elongation of ([\d.]+)',
                'vacuum_well': r'vacuum well depth of ([\d.]+)',
                'edge_rotational_transform': r'edge rotational transform of ([\d.]+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, prediction)
                if match:
                    try:
                        results[key] = float(match.group(1))
                    except ValueError:
                        pass
        
        elif format_type == 'qa_pairs':
            # Parse Q&A format
            qa_pairs = prediction.split('Q:')
            for qa in qa_pairs:
                if 'A:' in qa:
                    question, answer = qa.split('A:', 1)
                    answer = answer.strip().split('\n')[0]  # Take first line of answer
                    
                    # Try to extract numerical value
                    numbers = re.findall(r'[\d.]+', answer)
                    if numbers:
                        try:
                            value = float(numbers[0])
                            # Map question to metric name (simplified)
                            if 'aspect ratio' in question.lower() and 'computed' in question.lower():
                                results['aspect_ratio_computed'] = value
                            elif 'elongation' in question.lower():
                                results['max_elongation'] = value
                            elif 'vacuum well' in question.lower():
                                results['vacuum_well'] = value
                        except ValueError:
                            pass
        
        return results
    
    def evaluate_model(self, test_data_path: str, format_type: str = 'structured') -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data_path: Path to test data file
            format_type: Format type to use
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load test data (assuming it's in the same format as training data)
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_text = f.read()
        
        test_examples = [text.strip() for text in test_text.split('\n\n') if text.strip()]
        
        # Evaluate on a subset for speed
        n_eval = min(50, len(test_examples))
        test_examples = test_examples[:n_eval]
        
        predictions = []
        ground_truths = []
        
        for example in test_examples:
            # Extract input parameters and ground truth from the example
            # This is a simplified parsing - would need to be more robust
            if 'INPUT_PARAMETERS:' in example and 'OUTPUT_METRICS:' in example:
                input_section = example.split('OUTPUT_METRICS:')[0]
                output_section = example.split('OUTPUT_METRICS:')[1]
                
                # Parse input parameters (simplified)
                input_params = {}
                for line in input_section.split('\n'):
                    if ':' in line and any(param in line for param in ['aspect_ratio', 'elongation', 'rotational_transform']):
                        key, value = line.split(':', 1)
                        key = key.strip().replace(' ', '_')
                        try:
                            input_params[key] = float(value.strip())
                        except ValueError:
                            pass
                
                # Get prediction
                pred = self.predict_stellarator_metrics(input_params, format_type)
                predictions.append(pred)
                
                # Parse ground truth (simplified)
                gt = {}
                for line in output_section.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().replace(' ', '_')
                        try:
                            if value.strip() != 'N/A':
                                gt[key] = float(value.strip())
                        except ValueError:
                            pass
                ground_truths.append(gt)
        
        # Calculate metrics
        metrics = {}
        common_keys = set()
        for pred, gt in zip(predictions, ground_truths):
            common_keys.update(set(pred.keys()) & set(gt.keys()))
        
        for key in common_keys:
            pred_values = [pred.get(key, 0) for pred in predictions if key in pred]
            gt_values = [gt.get(key, 0) for gt in ground_truths if key in gt]
            
            if len(pred_values) > 0 and len(gt_values) > 0:
                # Calculate RMSE
                min_len = min(len(pred_values), len(gt_values))
                pred_values = pred_values[:min_len]
                gt_values = gt_values[:min_len]
                
                rmse = np.sqrt(np.mean((np.array(pred_values) - np.array(gt_values))**2))
                metrics[f'{key}_rmse'] = rmse
        
        self.logger.info(f"Evaluation completed on {len(test_examples)} examples")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value:.4f}")
        
        return metrics


def demo_language_model_training():
    """Demonstrate language model training on stellarator data."""
    
    print("🤖 Stellarator Language Model Training Demo")
    print("=" * 60)
    
    # Check if we have text data
    if not Path("./demo_text_data").exists():
        print("⚠️  No text data found. Running text conversion first...")
        from text_data_converter import demo_text_conversion
        demo_text_conversion()
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers library not available. Install with:")
        print("   pip install transformers datasets torch")
        return
    
    # Create language model
    model = StellaratorLanguageModel(model_name="gpt2", max_length=256)
    
    try:
        # Load text data
        train_texts, val_texts = model.load_text_data("./demo_text_data", "structured")
        
        # Train model (with minimal settings for demo)
        training_args = {
            'num_train_epochs': 1,
            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'warmup_steps': 10,
            'logging_steps': 10,
            'save_steps': 100,
            'eval_steps': 100,
        }
        
        model.train_model(
            train_texts, 
            val_texts, 
            output_dir="./demo_stellarator_lm",
            training_args=training_args
        )
        
        # Test inference
        print("\n🔮 Testing model inference...")
        print("-" * 40)
        
        test_params = {
            'aspect_ratio': 5.0,
            'elongation': 1.0,
            'rotational_transform': 0.4,
            'n_field_periods': 4,
            'triangularity': 0.0
        }
        
        prediction = model.predict_stellarator_metrics(test_params, 'structured')
        
        print("Input parameters:")
        for key, value in test_params.items():
            print(f"  {key}: {value}")
        
        print("\nPredicted metrics:")
        for key, value in prediction.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Language model training demo completed!")
        print(f"Model saved to: ./demo_stellarator_lm")
        
    except Exception as e:
        print(f"❌ Error in language model training: {e}")
        print("This might be due to insufficient data or computational resources.")


if __name__ == "__main__":
    demo_language_model_training()