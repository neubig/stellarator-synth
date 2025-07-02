#!/usr/bin/env python3
"""
Generate a synthetic dataset of stellarator configurations using VMEC++ and ConStellaration.
This script samples diverse stellarator configurations and computes their physical properties.
"""

import numpy as np
import pandas as pd
import os
import time
import json
from tqdm import tqdm
import concurrent.futures
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from constellaration.omnigeneity.omnigenity_field_sampling import SampleOmnigenousFieldAndTargetsSettings
    from constellaration.geometry import surface_rz_fourier
    from constellaration.mhd import ideal_mhd_parameters as ideal_mhd_parameters_module
    from constellaration.forward_model import forward_model, ConstellarationSettings
    from constellaration.utils import visualization
    from constellaration.data_generation.vmec_optimization_settings import OmnigenousFieldVmecOptimizationSettings
    from constellaration.data_generation import vmec_optimization
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please make sure constellaration and vmecpp are installed.")
    raise


def process_sample(sample, settings, index, output_dir):
    """
    Process a single omnigenous field sample to generate a stellarator configuration.
    
    Args:
        sample: The omnigenous field sample
        settings: Optimization settings
        index: Sample index for tracking
        output_dir: Directory to save outputs
        
    Returns:
        tuple: (input_features, output_features) or (None, None) if processing fails
    """
    try:
        logger.info(f"Processing sample {index}")
        start_time = time.time()
        
        # Optimize boundary for the omnigenous field
        boundary = vmec_optimization.optimize_boundary_omnigenity_vmec(
            sample, settings
        )
        
        # Run the forward model
        model_settings = ConstellarationSettings()
        metrics, equilibrium = forward_model(
            boundary=boundary,
            ideal_mhd_parameters=ideal_mhd_parameters_module.boundary_to_ideal_mhd_parameters(boundary),
            settings=model_settings
        )
        
        # Extract input features (boundary Fourier coefficients)
        input_features = {
            "n_field_periods": boundary.n_field_periods,
            "is_stellarator_symmetric": int(boundary.is_stellarator_symmetric),
        }
        
        # Add the Fourier coefficients
        for m in range(boundary.rbc.shape[0]):
            for n in range(boundary.rbc.shape[1]):
                input_features[f"rbc_{m}_{n}"] = boundary.rbc[m, n]
                input_features[f"zbs_{m}_{n}"] = boundary.zbs[m, n]
        
        # Extract output features (metrics)
        output_features = metrics.model_dump()
        
        # Save the boundary and visualization for reference
        boundary_file = os.path.join(output_dir, f"boundary_{index}.json")
        with open(boundary_file, "w") as f:
            f.write(boundary.model_dump_json(indent=2))
        
        # Optionally save a visualization
        if index % 100 == 0:  # Save visualization every 100 samples
            fig = visualization.plot_boundary(boundary)
            fig.write_image(os.path.join(output_dir, f"boundary_{index}.png"))
        
        # Save equilibrium data
        equilibrium_file = os.path.join(output_dir, f"equilibrium_{index}.json")
        with open(equilibrium_file, "w") as f:
            # Convert to dict and handle numpy arrays
            eq_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in equilibrium.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
            json.dump(eq_dict, f, indent=2)
        
        processing_time = time.time() - start_time
        logger.info(f"Completed sample {index} in {processing_time:.2f} seconds")
        
        return input_features, output_features
    
    except Exception as e:
        logger.error(f"Error in sample {index}: {e}", exc_info=True)
        return None, None


def generate_dataset(n_samples=10, output_dir="stellarator_dataset", max_workers=4, 
                    optimization_time=60, max_poloidal_mode=3, max_toroidal_mode=3):
    """
    Generate a dataset of stellarator configurations.
    
    Args:
        n_samples: Number of samples to generate
        output_dir: Directory to save the dataset
        max_workers: Maximum number of parallel workers
        optimization_time: Time limit for each optimization (seconds)
        max_poloidal_mode: Maximum poloidal mode number
        max_toroidal_mode: Maximum toroidal mode number
        
    Returns:
        tuple: (input_df, output_df) DataFrames containing the generated data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create settings for sampling
    logger.info(f"Generating {n_samples} samples with {max_workers} workers")
    sampler = SampleOmnigenousFieldAndTargetsSettings(n_samples=n_samples)
    samples = sampler.sample_omnigenous_fields_and_targets()
    
    # Storage for results
    input_data = []
    output_data = []
    
    # Optimization settings
    settings = OmnigenousFieldVmecOptimizationSettings(
        max_poloidal_mode=max_poloidal_mode,
        max_toroidal_mode=max_toroidal_mode,
        gradient_free_optimization_hypercube_bounds=0.33,
        gradient_free_max_time=optimization_time,
        verbose=False,
    )
    
    # Process each sample
    successful_samples = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, sample in enumerate(samples):
            # Submit the task to the executor
            future = executor.submit(
                process_sample, sample, settings, i, output_dir
            )
            futures.append(future)
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                sample_input, sample_output = future.result()
                if sample_input is not None and sample_output is not None:
                    input_data.append(sample_input)
                    output_data.append(sample_output)
                    successful_samples += 1
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
    
    # Save the dataset
    logger.info(f"Successfully processed {successful_samples} out of {n_samples} samples")
    
    if successful_samples > 0:
        input_df = pd.DataFrame(input_data)
        output_df = pd.DataFrame(output_data)
        
        input_df.to_csv(os.path.join(output_dir, "inputs.csv"), index=False)
        output_df.to_csv(os.path.join(output_dir, "outputs.csv"), index=False)
        
        # Save metadata
        metadata = {
            "n_samples": n_samples,
            "successful_samples": successful_samples,
            "max_poloidal_mode": max_poloidal_mode,
            "max_toroidal_mode": max_toroidal_mode,
            "optimization_time": optimization_time,
            "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset saved to {output_dir}")
        return input_df, output_df
    else:
        logger.error("No successful samples were generated")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stellarator dataset")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="stellarator_dataset", help="Output directory")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--optimization_time", type=int, default=60, help="Time limit for each optimization (seconds)")
    parser.add_argument("--max_poloidal_mode", type=int, default=3, help="Maximum poloidal mode number")
    parser.add_argument("--max_toroidal_mode", type=int, default=3, help="Maximum toroidal mode number")
    
    args = parser.parse_args()
    
    # Generate a dataset
    input_df, output_df = generate_dataset(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        optimization_time=args.optimization_time,
        max_poloidal_mode=args.max_poloidal_mode,
        max_toroidal_mode=args.max_toroidal_mode
    )