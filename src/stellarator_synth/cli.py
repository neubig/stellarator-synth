"""Command-line interface for stellarator-synth."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from stellarator_synth.data_generator import SyntheticDataConfig, SyntheticDataGenerator
from stellarator_synth.language_model_trainer import LanguageModelTrainer
from stellarator_synth.text_data_converter import TextDataConverter


def generate_data(args: argparse.Namespace) -> None:
    """Generate synthetic stellarator data."""
    config = SyntheticDataConfig(
        n_systematic_samples=args.n_systematic,
        n_lhs_samples=args.n_lhs,
        n_adaptive_samples=args.n_adaptive,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    
    generator = SyntheticDataGenerator(config)
    generator.generate_all_data()
    
    print(f"Generated {len(generator.successful_runs)} successful simulations")
    print(f"Failed simulations: {len(generator.failed_runs)}")


def convert_to_text(args: argparse.Namespace) -> None:
    """Convert simulation data to text format."""
    converter = TextDataConverter()
    
    if args.input_file.suffix == ".csv":
        import pandas as pd
        data = pd.read_csv(args.input_file).to_dict("records")
    else:
        import json
        with open(args.input_file) as f:
            data = json.load(f)
    
    converter.convert_to_text_dataset(
        data=data,
        output_dir=args.output_dir,
        formats=args.formats,
        train_split=args.train_split,
    )
    
    print(f"Converted data to text format in {args.output_dir}")


def train_language_model(args: argparse.Namespace) -> None:
    """Train a language model on stellarator data."""
    trainer = LanguageModelTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
    )
    
    trainer.train_from_text_files(
        train_file=args.train_file,
        val_file=args.val_file,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    print(f"Trained model saved to {args.output_dir}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stellarator synthetic data generation and language model training"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate data command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    gen_parser.add_argument("--n-systematic", type=int, default=1000, 
                           help="Number of systematic samples")
    gen_parser.add_argument("--n-lhs", type=int, default=2000,
                           help="Number of LHS samples")
    gen_parser.add_argument("--n-adaptive", type=int, default=1000,
                           help="Number of adaptive samples")
    gen_parser.add_argument("--n-workers", type=int, default=4,
                           help="Number of worker processes")
    gen_parser.add_argument("--batch-size", type=int, default=50,
                           help="Batch size for processing")
    gen_parser.add_argument("--output-dir", type=Path, default="./synthetic_data",
                           help="Output directory")
    
    # Convert to text command
    conv_parser = subparsers.add_parser("convert", help="Convert data to text format")
    conv_parser.add_argument("input_file", type=Path, help="Input data file")
    conv_parser.add_argument("--output-dir", type=Path, default="./text_data",
                            help="Output directory")
    conv_parser.add_argument("--formats", nargs="+", 
                            choices=["structured", "natural", "qa_pairs", "json_like", "code_like"],
                            default=["structured", "natural", "qa_pairs"],
                            help="Text formats to generate")
    conv_parser.add_argument("--train-split", type=float, default=0.8,
                            help="Training split ratio")
    
    # Train language model command
    train_parser = subparsers.add_parser("train", help="Train language model")
    train_parser.add_argument("train_file", type=Path, help="Training text file")
    train_parser.add_argument("--val-file", type=Path, help="Validation text file")
    train_parser.add_argument("--model-name", default="gpt2",
                             help="Base model name")
    train_parser.add_argument("--output-dir", type=Path, default="./trained_model",
                             help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=3,
                             help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=4,
                             help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, default=5e-5,
                             help="Learning rate")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generate_data(args)
    elif args.command == "convert":
        convert_to_text(args)
    elif args.command == "train":
        train_language_model(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()