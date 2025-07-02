#!/usr/bin/env python3
"""
Generate a small sample dataset for demonstration purposes.
This script runs a minimal version of the data generation pipeline.
"""

import os
import logging
import argparse
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sample_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate a small sample dataset")
    parser.add_argument("--n_samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="data/sample", help="Output directory")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument("--optimization_time", type=int, default=30, help="Time limit for optimization (seconds)")
    
    args = parser.parse_args()
    
    # Create output directories
    raw_data_dir = os.path.join(args.output_dir, "raw")
    ml_data_dir = os.path.join(args.output_dir, "ml")
    text_data_dir = os.path.join(args.output_dir, "text")
    
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(ml_data_dir, exist_ok=True)
    os.makedirs(text_data_dir, exist_ok=True)
    
    # Step 1: Generate synthetic data
    logger.info(f"Generating {args.n_samples} sample configurations")
    start_time = time.time()
    
    cmd = [
        "python", "src/generate_dataset.py",
        f"--n_samples={args.n_samples}",
        f"--output_dir={raw_data_dir}",
        f"--max_workers={args.max_workers}",
        f"--optimization_time={args.optimization_time}",
        "--max_poloidal_mode=2",
        "--max_toroidal_mode=2"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"Data generation failed: {process.stderr}")
        return
    
    logger.info(f"Data generation completed in {time.time() - start_time:.2f} seconds")
    
    # Check if the required files were generated
    input_csv = os.path.join(raw_data_dir, "inputs.csv")
    output_csv = os.path.join(raw_data_dir, "outputs.csv")
    
    if not os.path.exists(input_csv) or not os.path.exists(output_csv):
        logger.error("Required CSV files were not generated")
        return
    
    # Step 2: Process the data for ML
    logger.info("Processing data for machine learning")
    cmd = [
        "python", "src/process_dataset.py",
        f"--input_csv={input_csv}",
        f"--output_csv={output_csv}",
        f"--output_dir={ml_data_dir}"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"Data processing failed: {process.stderr}")
        return
    
    logger.info("Data processing completed")
    
    # Step 3: Convert to text format
    logger.info("Converting data to text format")
    cmd = [
        "python", "src/convert_to_text.py",
        f"--input_csv={input_csv}",
        f"--output_csv={output_csv}",
        f"--boundary_dir={raw_data_dir}",
        f"--equilibrium_dir={raw_data_dir}",
        f"--output_dir={text_data_dir}",
        "--format=prompt"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"Text conversion failed: {process.stderr}")
        return
    
    logger.info("Text conversion completed")
    
    # Print summary
    logger.info("\nSample Dataset Generation Summary:")
    logger.info(f"- Raw data: {raw_data_dir}")
    logger.info(f"- ML data: {ml_data_dir}")
    logger.info(f"- Text data: {text_data_dir}")
    logger.info("\nTo view a sample text description:")
    logger.info(f"  cat {text_data_dir}/all_samples.txt")
    
    # Create a README file in the output directory
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Sample Stellarator Dataset\n\n")
        f.write(f"This directory contains a small sample dataset of {args.n_samples} stellarator configurations.\n\n")
        f.write("## Directory Structure\n\n")
        f.write("- `raw/`: Raw data from the simulator\n")
        f.write("  - `inputs.csv`: Input features (boundary parameters)\n")
        f.write("  - `outputs.csv`: Output features (physics metrics)\n")
        f.write("  - `boundary_*.json`: Boundary surface descriptions\n")
        f.write("  - `equilibrium_*.json`: Equilibrium data\n\n")
        f.write("- `ml/`: Processed data for machine learning\n")
        f.write("  - `X_train.npy`, `y_train.npy`: Training data\n")
        f.write("  - `X_val.npy`, `y_val.npy`: Validation data\n")
        f.write("  - `X_test.npy`, `y_test.npy`: Test data\n\n")
        f.write("- `text/`: Text format data for language models\n")
        f.write("  - `all_samples.txt`: Combined text descriptions\n")
        f.write("  - `prompts.txt`: Input prompts for language model\n")
        f.write("  - `completions.txt`: Target completions for language model\n\n")
        f.write("## Generation Details\n\n")
        f.write(f"- Number of samples: {args.n_samples}\n")
        f.write(f"- Optimization time: {args.optimization_time} seconds per sample\n")
        f.write(f"- Max poloidal mode: 2\n")
        f.write(f"- Max toroidal mode: 2\n")
        f.write(f"- Generation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logger.info(f"README file created: {readme_path}")
    logger.info("Sample dataset generation completed successfully!")


if __name__ == "__main__":
    main()