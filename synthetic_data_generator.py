#!/usr/bin/env python3
"""
Synthetic Data Generator for Stellarator Simulator ML Surrogate

This script demonstrates how to generate large amounts of synthetic training data
using the open-source VMEC++ simulator and ConStellaration framework.
"""

import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd
from scipy.stats import qmc
import pydantic

# ConStellaration imports (would need to be installed)
try:
    from constellaration import forward_model, initial_guess
    from constellaration.geometry import surface_rz_fourier
    from constellaration.mhd import ideal_mhd_parameters
    CONSTELLARATION_AVAILABLE = True
except ImportError:
    CONSTELLARATION_AVAILABLE = False
    print("ConStellaration not available. This is a demonstration script.")


class SyntheticDataConfig(pydantic.BaseModel):
    """Configuration for synthetic data generation."""
    
    # Parameter ranges for systematic exploration
    aspect_ratio_range: Tuple[float, float] = (2.5, 8.0)
    elongation_range: Tuple[float, float] = (0.3, 2.0)
    rotational_transform_range: Tuple[float, float] = (0.2, 0.8)
    n_field_periods_options: List[int] = [2, 3, 4, 5, 6]
    triangularity_range: Tuple[float, float] = (-0.3, 0.5)
    
    # Sampling parameters
    n_systematic_samples: int = 10000
    n_lhs_samples: int = 25000
    n_adaptive_samples: int = 15000
    
    # Computational parameters
    n_workers: int = 8
    batch_size: int = 100
    max_retries: int = 3
    
    # Output parameters
    output_dir: str = "./synthetic_stellarator_data"
    save_frequency: int = 1000
    
    # Fidelity settings
    use_high_fidelity: bool = False
    include_qi_analysis: bool = True
    include_turbulent_transport: bool = True


class SyntheticDataGenerator:
    """Main class for generating synthetic stellarator data."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'generation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.successful_runs = []
        self.failed_runs = []
        
    def generate_systematic_parameters(self) -> List[Dict[str, Any]]:
        """Generate parameters using systematic grid sampling."""
        self.logger.info("Generating systematic parameter grid...")
        
        # Define parameter grids
        aspect_ratios = np.linspace(*self.config.aspect_ratio_range, 9)
        elongations = np.linspace(*self.config.elongation_range, 8)
        rotational_transforms = np.linspace(*self.config.rotational_transform_range, 7)
        triangularities = np.linspace(*self.config.triangularity_range, 6)
        
        parameters = []
        for ar, el, rt, nfp, tri in itertools.product(
            aspect_ratios, elongations, rotational_transforms, 
            self.config.n_field_periods_options, triangularities
        ):
            parameters.append({
                'aspect_ratio': float(ar),
                'elongation': float(el),
                'rotational_transform': float(rt),
                'n_field_periods': int(nfp),
                'triangularity': float(tri),
                'generation_method': 'systematic'
            })
            
            if len(parameters) >= self.config.n_systematic_samples:
                break
                
        self.logger.info(f"Generated {len(parameters)} systematic parameter sets")
        return parameters
    
    def generate_lhs_parameters(self) -> List[Dict[str, Any]]:
        """Generate parameters using Latin Hypercube Sampling."""
        self.logger.info("Generating LHS parameter samples...")
        
        # Setup LHS sampler
        sampler = qmc.LatinHypercube(d=4)  # 4 continuous parameters
        samples = sampler.random(n=self.config.n_lhs_samples)
        
        parameters = []
        for sample in samples:
            # Scale samples to parameter ranges
            aspect_ratio = self.config.aspect_ratio_range[0] + sample[0] * (
                self.config.aspect_ratio_range[1] - self.config.aspect_ratio_range[0]
            )
            elongation = self.config.elongation_range[0] + sample[1] * (
                self.config.elongation_range[1] - self.config.elongation_range[0]
            )
            rotational_transform = self.config.rotational_transform_range[0] + sample[2] * (
                self.config.rotational_transform_range[1] - self.config.rotational_transform_range[0]
            )
            triangularity = self.config.triangularity_range[0] + sample[3] * (
                self.config.triangularity_range[1] - self.config.triangularity_range[0]
            )
            
            # Randomly select discrete parameters
            n_field_periods = np.random.choice(self.config.n_field_periods_options)
            
            parameters.append({
                'aspect_ratio': float(aspect_ratio),
                'elongation': float(elongation),
                'rotational_transform': float(rotational_transform),
                'n_field_periods': int(n_field_periods),
                'triangularity': float(triangularity),
                'generation_method': 'lhs'
            })
            
        self.logger.info(f"Generated {len(parameters)} LHS parameter sets")
        return parameters
    
    def generate_adaptive_parameters(self, existing_data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Generate parameters using adaptive sampling (simplified version)."""
        self.logger.info("Generating adaptive parameter samples...")
        
        # For this demonstration, we'll use random sampling with bias toward
        # regions that might be underrepresented
        parameters = []
        
        for _ in range(self.config.n_adaptive_samples):
            # Add some bias toward extreme parameter values
            if np.random.random() < 0.3:  # 30% chance for extreme values
                aspect_ratio = np.random.choice([
                    np.random.uniform(2.5, 3.5),  # Low aspect ratio
                    np.random.uniform(6.0, 8.0)   # High aspect ratio
                ])
                elongation = np.random.choice([
                    np.random.uniform(0.3, 0.6),  # Low elongation
                    np.random.uniform(1.5, 2.0)   # High elongation
                ])
            else:
                aspect_ratio = np.random.uniform(*self.config.aspect_ratio_range)
                elongation = np.random.uniform(*self.config.elongation_range)
            
            rotational_transform = np.random.uniform(*self.config.rotational_transform_range)
            triangularity = np.random.uniform(*self.config.triangularity_range)
            n_field_periods = np.random.choice(self.config.n_field_periods_options)
            
            parameters.append({
                'aspect_ratio': float(aspect_ratio),
                'elongation': float(elongation),
                'rotational_transform': float(rotational_transform),
                'n_field_periods': int(n_field_periods),
                'triangularity': float(triangularity),
                'generation_method': 'adaptive'
            })
            
        self.logger.info(f"Generated {len(parameters)} adaptive parameter sets")
        return parameters
    
    def run_single_simulation(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run a single VMEC++ simulation with given parameters."""
        if not CONSTELLARATION_AVAILABLE:
            # Simulate the computation for demonstration
            time.sleep(0.1)  # Simulate computation time
            
            # Generate mock results
            if np.random.random() < 0.95:  # 95% success rate
                return {
                    **params,
                    'success': True,
                    'aspect_ratio_computed': params['aspect_ratio'] + np.random.normal(0, 0.1),
                    'max_elongation': params['elongation'] + np.random.normal(0, 0.05),
                    'edge_rotational_transform': params['rotational_transform'] + np.random.normal(0, 0.02),
                    'qi_residual': np.random.exponential(0.1),
                    'vacuum_well': np.random.normal(0.5, 0.1),
                    'computation_time': np.random.uniform(0.5, 2.0)
                }
            else:
                return {
                    **params,
                    'success': False,
                    'error_message': 'VMEC convergence failed'
                }
        
        try:
            # Create boundary from parameters
            boundary = initial_guess.generate_rotating_ellipse(
                aspect_ratio=params['aspect_ratio'],
                elongation=params['elongation'],
                rotational_transform=params['rotational_transform'],
                n_field_periods=params['n_field_periods']
            )
            
            # Add triangularity if specified
            if params.get('triangularity', 0) != 0:
                # This would require additional boundary modification
                pass
            
            # Setup simulation settings
            if self.config.use_high_fidelity:
                settings = forward_model.ConstellarationSettings.default_high_fidelity()
            else:
                settings = forward_model.ConstellarationSettings()
            
            if not self.config.include_qi_analysis:
                settings.qi_settings = None
                settings.boozer_preset_settings = None
            
            if not self.config.include_turbulent_transport:
                settings.turbulent_settings = None
            
            # Run simulation
            start_time = time.time()
            metrics, equilibrium = forward_model.forward_model(
                boundary=boundary,
                settings=settings
            )
            computation_time = time.time() - start_time
            
            # Extract results
            result = {
                **params,
                'success': True,
                'aspect_ratio_computed': float(metrics.aspect_ratio),
                'max_elongation': float(metrics.max_elongation),
                'edge_rotational_transform': float(metrics.edge_rotational_transform_over_n_field_periods),
                'axis_rotational_transform': float(metrics.axis_rotational_transform_over_n_field_periods),
                'vacuum_well': float(metrics.vacuum_well),
                'average_triangularity': float(metrics.average_triangularity),
                'axis_magnetic_mirror_ratio': float(metrics.axis_magnetic_mirror_ratio),
                'edge_magnetic_mirror_ratio': float(metrics.edge_magnetic_mirror_ratio),
                'min_normalized_magnetic_gradient_scale_length': float(metrics.minimum_normalized_magnetic_gradient_scale_length),
                'computation_time': computation_time
            }
            
            if metrics.qi is not None:
                result['qi_residual'] = float(metrics.qi)
            
            if metrics.flux_compression_in_regions_of_bad_curvature is not None:
                result['flux_compression_bad_curvature'] = float(metrics.flux_compression_in_regions_of_bad_curvature)
            
            return result
            
        except Exception as e:
            return {
                **params,
                'success': False,
                'error_message': str(e)
            }
    
    def run_batch_simulations(self, parameter_batch: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Run a batch of simulations."""
        successful = []
        failed = []
        
        for params in parameter_batch:
            for attempt in range(self.config.max_retries):
                result = self.run_single_simulation(params)
                if result is not None:
                    if result['success']:
                        successful.append(result)
                    else:
                        failed.append(result)
                    break
            else:
                # All retries failed
                failed.append({
                    **params,
                    'success': False,
                    'error_message': 'Max retries exceeded'
                })
        
        return successful, failed
    
    def save_data(self, data: List[Dict], filename: str):
        """Save data to file."""
        df = pd.DataFrame(data)
        
        # Save as both CSV and JSON
        csv_path = self.output_dir / f"{filename}.csv"
        json_path = self.output_dir / f"{filename}.json"
        
        df.to_csv(csv_path, index=False)
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved {len(data)} records to {csv_path} and {json_path}")
    
    def generate_all_data(self):
        """Generate all synthetic data."""
        self.logger.info("Starting synthetic data generation...")
        
        # Generate all parameter sets
        all_parameters = []
        all_parameters.extend(self.generate_systematic_parameters())
        all_parameters.extend(self.generate_lhs_parameters())
        all_parameters.extend(self.generate_adaptive_parameters())
        
        self.logger.info(f"Total parameter sets to evaluate: {len(all_parameters)}")
        
        # Process in batches
        total_batches = (len(all_parameters) + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(all_parameters))
            batch = all_parameters[start_idx:end_idx]
            
            self.logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} simulations)")
            
            # Run batch
            successful, failed = self.run_batch_simulations(batch)
            
            self.successful_runs.extend(successful)
            self.failed_runs.extend(failed)
            
            # Save intermediate results
            if (batch_idx + 1) % (self.config.save_frequency // self.config.batch_size) == 0:
                self.save_data(self.successful_runs, f"successful_runs_batch_{batch_idx + 1}")
                self.save_data(self.failed_runs, f"failed_runs_batch_{batch_idx + 1}")
            
            # Log progress
            success_rate = len(successful) / len(batch) * 100
            total_success_rate = len(self.successful_runs) / (len(self.successful_runs) + len(self.failed_runs)) * 100
            
            self.logger.info(f"Batch {batch_idx + 1} success rate: {success_rate:.1f}%")
            self.logger.info(f"Overall success rate: {total_success_rate:.1f}%")
        
        # Save final results
        self.save_data(self.successful_runs, "successful_runs_final")
        self.save_data(self.failed_runs, "failed_runs_final")
        
        # Generate summary statistics
        self.generate_summary_statistics()
        
        self.logger.info("Synthetic data generation completed!")
    
    def generate_summary_statistics(self):
        """Generate summary statistics for the generated data."""
        if not self.successful_runs:
            self.logger.warning("No successful runs to analyze")
            return
        
        df = pd.DataFrame(self.successful_runs)
        
        # Basic statistics
        summary = {
            'total_simulations': len(self.successful_runs) + len(self.failed_runs),
            'successful_simulations': len(self.successful_runs),
            'failed_simulations': len(self.failed_runs),
            'success_rate': len(self.successful_runs) / (len(self.successful_runs) + len(self.failed_runs)),
            'parameter_statistics': {},
            'metric_statistics': {}
        }
        
        # Parameter coverage
        for param in ['aspect_ratio', 'elongation', 'rotational_transform', 'triangularity']:
            if param in df.columns:
                summary['parameter_statistics'][param] = {
                    'min': float(df[param].min()),
                    'max': float(df[param].max()),
                    'mean': float(df[param].mean()),
                    'std': float(df[param].std())
                }
        
        # Metric statistics
        metric_columns = [col for col in df.columns if col.endswith('_computed') or 'residual' in col or 'well' in col]
        for metric in metric_columns:
            if metric in df.columns:
                summary['metric_statistics'][metric] = {
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max()),
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std())
                }
        
        # Save summary
        summary_path = self.output_dir / "generation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Generated summary statistics: {summary_path}")
        self.logger.info(f"Success rate: {summary['success_rate']:.1%}")


def main():
    """Main function to run synthetic data generation."""
    
    # Configuration
    config = SyntheticDataConfig(
        n_systematic_samples=1000,   # Reduced for demonstration
        n_lhs_samples=2000,          # Reduced for demonstration
        n_adaptive_samples=1000,     # Reduced for demonstration
        n_workers=4,
        batch_size=50,
        output_dir="./demo_synthetic_data"
    )
    
    # Create generator
    generator = SyntheticDataGenerator(config)
    
    # Generate data
    generator.generate_all_data()
    
    print(f"\nSynthetic data generation completed!")
    print(f"Results saved to: {generator.output_dir}")
    print(f"Successful simulations: {len(generator.successful_runs)}")
    print(f"Failed simulations: {len(generator.failed_runs)}")
    
    if generator.successful_runs:
        success_rate = len(generator.successful_runs) / (len(generator.successful_runs) + len(generator.failed_runs))
        print(f"Success rate: {success_rate:.1%}")


if __name__ == "__main__":
    main()