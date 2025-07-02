#!/usr/bin/env python3
"""
Generate example stellarator data for demonstration purposes.
This script creates synthetic data that mimics the structure of real stellarator data.
"""

import os
import json
import numpy as np
import pandas as pd
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("example_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_boundary(n_field_periods=5, max_poloidal_mode=3, max_toroidal_mode=3):
    """Generate a synthetic stellarator boundary."""
    # Create Fourier coefficients
    rbc = np.zeros((max_poloidal_mode + 1, max_toroidal_mode + 1))
    zbs = np.zeros((max_poloidal_mode + 1, max_toroidal_mode + 1))
    
    # Set R00 to 1.0 (major radius)
    rbc[0, 0] = 1.0
    
    # Add some non-zero coefficients to make an interesting shape
    rbc[1, 0] = 0.1    # Vertical elongation
    rbc[0, 1] = 0.05   # Toroidal effect
    rbc[1, 1] = 0.02   # Helical component
    rbc[2, 2] = 0.01   # Higher order shaping
    
    zbs[1, 0] = 0.1    # Vertical elongation
    zbs[0, 1] = 0.05   # Toroidal effect
    zbs[1, 1] = 0.03   # Helical component
    zbs[2, 2] = 0.01   # Higher order shaping
    
    # Create boundary data
    boundary = {
        "n_field_periods": n_field_periods,
        "is_stellarator_symmetric": True,
        "rbc": rbc.tolist(),
        "zbs": zbs.tolist()
    }
    
    return boundary


def generate_equilibrium(boundary):
    """Generate synthetic equilibrium data based on a boundary."""
    # Extract parameters from boundary
    n_field_periods = boundary["n_field_periods"]
    
    # Create equilibrium data
    equilibrium = {
        "aspect": 5.0 + np.random.normal(0, 0.2),
        "volume": 10.0 + np.random.normal(0, 1.0),
        "b0": 1.0 + np.random.normal(0, 0.05),
        "rmax_surf": 1.1 + np.random.normal(0, 0.02),
        "rmin_surf": 0.9 + np.random.normal(0, 0.02),
        "zmax_surf": 0.2 + np.random.normal(0, 0.01),
        "n_field_periods": n_field_periods,
        "mpol": len(boundary["rbc"]),
        "ntor": len(boundary["rbc"][0]),
        "iota_full": [0.2 + 0.1 * s + 0.05 * s**2 + np.random.normal(0, 0.01) for s in np.linspace(0, 1, 10)],
        "vacuum_well": 0.01 + np.random.normal(0, 0.002),
        "magnetic_well": 0.02 + np.random.normal(0, 0.003),
        "max_elongation": 2.0 + np.random.normal(0, 0.1),
        "min_curvature_radius": 0.1 + np.random.normal(0, 0.01),
        "max_curvature": 5.0 + np.random.normal(0, 0.5),
        "coil_complexity": 8.0 + np.random.normal(0, 1.0),
        "bootstrap_current": 0.05 + np.random.normal(0, 0.01),
        "neoclassical_transport": 0.2 + np.random.normal(0, 0.05),
        "effective_ripple": 0.01 + np.random.normal(0, 0.002)
    }
    
    return equilibrium


def extract_input_features(boundary):
    """Extract input features from a boundary."""
    input_features = {
        "n_field_periods": boundary["n_field_periods"],
        "is_stellarator_symmetric": int(boundary["is_stellarator_symmetric"]),
    }
    
    # Add the Fourier coefficients
    rbc = np.array(boundary["rbc"])
    zbs = np.array(boundary["zbs"])
    
    for m in range(rbc.shape[0]):
        for n in range(rbc.shape[1]):
            input_features[f"rbc_{m}_{n}"] = rbc[m, n]
            input_features[f"zbs_{m}_{n}"] = zbs[m, n]
    
    return input_features


def extract_output_features(equilibrium):
    """Extract output features from an equilibrium."""
    # Select relevant metrics
    output_features = {
        "aspect_ratio": equilibrium["aspect"],
        "volume": equilibrium["volume"],
        "b0": equilibrium["b0"],
        "rmax_surf": equilibrium["rmax_surf"],
        "rmin_surf": equilibrium["rmin_surf"],
        "zmax_surf": equilibrium["zmax_surf"],
        "vacuum_well": equilibrium["vacuum_well"],
        "magnetic_well": equilibrium["magnetic_well"],
        "max_elongation": equilibrium["max_elongation"],
        "min_curvature_radius": equilibrium["min_curvature_radius"],
        "max_curvature": equilibrium["max_curvature"],
        "coil_complexity": equilibrium["coil_complexity"],
        "bootstrap_current": equilibrium["bootstrap_current"],
        "neoclassical_transport": equilibrium["neoclassical_transport"],
        "effective_ripple": equilibrium["effective_ripple"]
    }
    
    return output_features


def generate_example_dataset(n_samples=10, output_dir="data/example"):
    """Generate a synthetic dataset of stellarator configurations."""
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Generating {n_samples} example stellarator configurations")
    
    # Storage for results
    input_data = []
    output_data = []
    
    # Custom JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    for i in range(n_samples):
        # Generate boundary
        n_field_periods = np.random.choice([2, 3, 4, 5])
        max_poloidal_mode = 3
        max_toroidal_mode = 3
        
        boundary = generate_boundary(
            n_field_periods=int(n_field_periods),  # Convert to Python int
            max_poloidal_mode=max_poloidal_mode,
            max_toroidal_mode=max_toroidal_mode
        )
        
        # Generate equilibrium
        equilibrium = generate_equilibrium(boundary)
        
        # Extract features
        input_features = extract_input_features(boundary)
        output_features = extract_output_features(equilibrium)
        
        # Save boundary and equilibrium
        boundary_file = os.path.join(output_dir, f"boundary_{i}.json")
        with open(boundary_file, "w") as f:
            json.dump(boundary, f, indent=2, cls=NumpyEncoder)
        
        equilibrium_file = os.path.join(output_dir, f"equilibrium_{i}.json")
        with open(equilibrium_file, "w") as f:
            json.dump(equilibrium, f, indent=2, cls=NumpyEncoder)
        
        # Store features
        input_data.append(input_features)
        output_data.append(output_features)
    
    # Save features to CSV
    input_df = pd.DataFrame(input_data)
    output_df = pd.DataFrame(output_data)
    
    input_csv = os.path.join(output_dir, "inputs.csv")
    output_csv = os.path.join(output_dir, "outputs.csv")
    
    input_df.to_csv(input_csv, index=False)
    output_df.to_csv(output_csv, index=False)
    
    # Save metadata
    metadata = {
        "n_samples": n_samples,
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Synthetic stellarator data for demonstration purposes"
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Example dataset saved to {output_dir}")
    logger.info(f"Input features: {input_csv}")
    logger.info(f"Output features: {output_csv}")
    
    return input_df, output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate example stellarator data")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="data/example", help="Output directory")
    
    args = parser.parse_args()
    
    generate_example_dataset(
        n_samples=args.n_samples,
        output_dir=args.output_dir
    )