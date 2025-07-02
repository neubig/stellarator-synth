#!/usr/bin/env python3
"""
Convert stellarator data into text format for language model training.
This script transforms numerical data into structured text descriptions.
"""

import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("text_conversion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def format_number(value):
    """Format a number for text representation."""
    if isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, (float, np.floating)):
        # Format with appropriate precision
        if abs(value) < 0.001:
            return f"{value:.8f}"
        elif abs(value) < 0.1:
            return f"{value:.6f}"
        else:
            return f"{value:.4f}"
    else:
        return str(value)


def boundary_to_text(boundary_file):
    """Convert a boundary JSON file to text description."""
    with open(boundary_file, 'r') as f:
        boundary_data = json.load(f)
    
    text = "STELLARATOR BOUNDARY DESCRIPTION\n"
    text += "================================\n\n"
    
    # Basic configuration
    text += f"Field periods: {boundary_data.get('n_field_periods', 'N/A')}\n"
    text += f"Stellarator symmetric: {boundary_data.get('is_stellarator_symmetric', 'N/A')}\n\n"
    
    # Fourier coefficients
    text += "Fourier coefficients (RBC):\n"
    rbc = np.array(boundary_data.get('rbc', []))
    if len(rbc) > 0:
        for m in range(rbc.shape[0]):
            for n in range(rbc.shape[1]):
                if abs(rbc[m][n]) > 1e-6:  # Only include non-zero coefficients
                    text += f"  m={m}, n={n}: {format_number(rbc[m][n])}\n"
    
    text += "\nFourier coefficients (ZBS):\n"
    zbs = np.array(boundary_data.get('zbs', []))
    if len(zbs) > 0:
        for m in range(zbs.shape[0]):
            for n in range(zbs.shape[1]):
                if abs(zbs[m][n]) > 1e-6:  # Only include non-zero coefficients
                    text += f"  m={m}, n={n}: {format_number(zbs[m][n])}\n"
    
    return text


def equilibrium_to_text(equilibrium_file):
    """Convert an equilibrium JSON file to text description."""
    with open(equilibrium_file, 'r') as f:
        eq_data = json.load(f)
    
    text = "STELLARATOR EQUILIBRIUM PROPERTIES\n"
    text += "==================================\n\n"
    
    # Basic properties
    properties = [
        ("aspect", "Aspect ratio"),
        ("volume", "Plasma volume"),
        ("b0", "Reference magnetic field strength"),
        ("rmax_surf", "Maximum major radius"),
        ("rmin_surf", "Minimum major radius"),
        ("zmax_surf", "Maximum height"),
        ("n_field_periods", "Number of field periods"),
        ("mpol", "Number of poloidal modes"),
        ("ntor", "Number of toroidal modes")
    ]
    
    for key, label in properties:
        if key in eq_data:
            text += f"{label}: {format_number(eq_data[key])}\n"
    
    # Iota profile (rotational transform)
    if "iota_full" in eq_data:
        text += "\nRotational transform profile:\n"
        iota = eq_data["iota_full"]
        s_values = np.linspace(0, 1, len(iota))
        for i, (s, iota_val) in enumerate(zip(s_values, iota)):
            if i % max(1, len(iota) // 10) == 0:  # Sample ~10 points
                text += f"  s={format_number(s)}: iota={format_number(iota_val)}\n"
    
    return text


def metrics_to_text(input_row, output_row):
    """Convert metrics data to text description."""
    text = "STELLARATOR PERFORMANCE METRICS\n"
    text += "===============================\n\n"
    
    # Input features that are not Fourier coefficients
    for col in input_row.index:
        if not (col.startswith('rbc_') or col.startswith('zbs_')):
            text += f"{col}: {format_number(input_row[col])}\n"
    
    text += "\nPerformance metrics:\n"
    for col in output_row.index:
        text += f"{col}: {format_number(output_row[col])}\n"
    
    return text


def generate_text_dataset(input_csv, output_csv, boundary_dir, equilibrium_dir, output_dir, 
                         format_type="separate", max_samples=None):
    """
    Generate a text dataset from stellarator data.
    
    Args:
        input_csv: Path to input features CSV
        output_csv: Path to output features CSV
        boundary_dir: Directory containing boundary JSON files
        equilibrium_dir: Directory containing equilibrium JSON files
        output_dir: Directory to save text files
        format_type: Format type ("separate", "combined", or "prompt")
        max_samples: Maximum number of samples to process (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV data
    inputs_df = pd.read_csv(input_csv)
    outputs_df = pd.read_csv(output_csv)
    
    # Get list of boundary and equilibrium files
    boundary_files = sorted(glob.glob(os.path.join(boundary_dir, "boundary_*.json")))
    equilibrium_files = sorted(glob.glob(os.path.join(equilibrium_dir, "equilibrium_*.json")))
    
    # Determine number of samples to process
    n_samples = min(len(inputs_df), len(outputs_df))
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)
    
    logger.info(f"Converting {n_samples} samples to text format")
    
    # Create a combined text file for all samples
    all_text_file = os.path.join(output_dir, "all_samples.txt")
    prompt_file = os.path.join(output_dir, "prompts.txt")
    completion_file = os.path.join(output_dir, "completions.txt")
    
    with open(all_text_file, 'w') as all_file, \
         open(prompt_file, 'w') as prompt_f, \
         open(completion_file, 'w') as completion_f:
        
        for i in tqdm(range(n_samples)):
            # Get corresponding files
            boundary_file = os.path.join(boundary_dir, f"boundary_{i}.json")
            equilibrium_file = os.path.join(equilibrium_dir, f"equilibrium_{i}.json")
            
            # Skip if files don't exist
            if not os.path.exists(boundary_file) or not os.path.exists(equilibrium_file):
                logger.warning(f"Missing files for sample {i}, skipping")
                continue
            
            # Convert data to text
            boundary_text = boundary_to_text(boundary_file)
            equilibrium_text = equilibrium_to_text(equilibrium_file)
            metrics_text = metrics_to_text(inputs_df.iloc[i], outputs_df.iloc[i])
            
            # Save individual files if requested
            if format_type == "separate":
                with open(os.path.join(output_dir, f"sample_{i}_boundary.txt"), 'w') as f:
                    f.write(boundary_text)
                with open(os.path.join(output_dir, f"sample_{i}_equilibrium.txt"), 'w') as f:
                    f.write(equilibrium_text)
                with open(os.path.join(output_dir, f"sample_{i}_metrics.txt"), 'w') as f:
                    f.write(metrics_text)
            
            # Combined format
            combined_text = f"SAMPLE {i}\n\n{boundary_text}\n\n{equilibrium_text}\n\n{metrics_text}\n\n"
            combined_text += "=" * 80 + "\n\n"
            
            # Write to combined file
            all_file.write(combined_text)
            
            # Create prompt-completion pairs for language model fine-tuning
            if format_type == "prompt":
                # Prompt: boundary description
                # Completion: equilibrium and metrics
                prompt = f"Given the following stellarator boundary description, predict the equilibrium properties and performance metrics:\n\n{boundary_text}\n"
                completion = f"{equilibrium_text}\n\n{metrics_text}"
                
                prompt_f.write(prompt + "\n")
                completion_f.write(completion + "\n")
    
    logger.info(f"Text dataset saved to {output_dir}")
    logger.info(f"Combined text file: {all_text_file}")
    if format_type == "prompt":
        logger.info(f"Prompt file: {prompt_file}")
        logger.info(f"Completion file: {completion_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert stellarator data to text format")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input features CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output features CSV")
    parser.add_argument("--boundary_dir", type=str, required=True, help="Directory with boundary JSON files")
    parser.add_argument("--equilibrium_dir", type=str, required=True, help="Directory with equilibrium JSON files")
    parser.add_argument("--output_dir", type=str, default="data/text_dataset", help="Directory to save text files")
    parser.add_argument("--format", type=str, choices=["separate", "combined", "prompt"], default="combined",
                       help="Format type: separate files, combined file, or prompt-completion pairs")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    generate_text_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        boundary_dir=args.boundary_dir,
        equilibrium_dir=args.equilibrium_dir,
        output_dir=args.output_dir,
        format_type=args.format,
        max_samples=args.max_samples
    )