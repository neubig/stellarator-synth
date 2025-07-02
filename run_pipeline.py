#!/usr/bin/env python3
"""
Run the complete stellarator synthetic data pipeline:
1. Generate synthetic data
2. Process the data for machine learning
3. Train a surrogate model
"""

import os
import argparse
import logging
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_command(command):
    """Run a shell command and log the output."""
    logger.info(f"Running command: {command}")
    start_time = time.time()
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream the output
    for line in process.stdout:
        logger.info(line.strip())
    
    process.wait()
    execution_time = time.time() - start_time
    
    if process.returncode != 0:
        logger.error(f"Command failed with return code {process.returncode}")
        return False
    
    logger.info(f"Command completed in {execution_time:.2f} seconds")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the stellarator synthetic data pipeline")
    
    # Data generation parameters
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--raw_data_dir", type=str, default="data/raw_dataset", help="Directory for raw data")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--optimization_time", type=int, default=60, help="Time limit for each optimization (seconds)")
    
    # Data processing parameters
    parser.add_argument("--ml_data_dir", type=str, default="data/ml_dataset", help="Directory for processed ML data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--val_size", type=float, default=0.15, help="Fraction of data for validation")
    
    # Model training parameters
    parser.add_argument("--model_dir", type=str, default="models/surrogate_model", help="Directory for model output")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--layer_sizes", type=str, default="256,128,64", help="Hidden layer sizes (comma-separated)")
    
    # Pipeline control
    parser.add_argument("--skip_generation", action="store_true", help="Skip data generation step")
    parser.add_argument("--skip_processing", action="store_true", help="Skip data processing step")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training step")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.raw_data_dir, exist_ok=True)
    os.makedirs(args.ml_data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Step 1: Generate synthetic data
    if not args.skip_generation:
        logger.info("=== Step 1: Generating synthetic data ===")
        command = (
            f"python src/generate_dataset.py "
            f"--n_samples {args.n_samples} "
            f"--output_dir {args.raw_data_dir} "
            f"--max_workers {args.max_workers} "
            f"--optimization_time {args.optimization_time}"
        )
        if not run_command(command):
            logger.error("Data generation failed. Exiting pipeline.")
            return
    else:
        logger.info("Skipping data generation step.")
    
    # Step 2: Process the data
    if not args.skip_processing:
        logger.info("=== Step 2: Processing data for machine learning ===")
        command = (
            f"python src/process_dataset.py "
            f"--input_csv {args.raw_data_dir}/inputs.csv "
            f"--output_csv {args.raw_data_dir}/outputs.csv "
            f"--output_dir {args.ml_data_dir} "
            f"--test_size {args.test_size} "
            f"--val_size {args.val_size}"
        )
        if not run_command(command):
            logger.error("Data processing failed. Exiting pipeline.")
            return
    else:
        logger.info("Skipping data processing step.")
    
    # Step 3: Train the model
    if not args.skip_training:
        logger.info("=== Step 3: Training surrogate model ===")
        command = (
            f"python src/train_model.py "
            f"--data_dir {args.ml_data_dir} "
            f"--output_dir {args.model_dir} "
            f"--batch_size {args.batch_size} "
            f"--epochs {args.epochs} "
            f"--patience {args.patience} "
            f"--layer_sizes {args.layer_sizes}"
        )
        if not run_command(command):
            logger.error("Model training failed.")
            return
    else:
        logger.info("Skipping model training step.")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()