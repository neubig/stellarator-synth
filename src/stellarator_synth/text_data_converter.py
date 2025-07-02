#!/usr/bin/env python3
"""
Text Data Converter for Stellarator Language Model Training

This module converts stellarator simulation data into structured text format
suitable for language model training.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np


class StellaratorTextConverter:
    """Converts stellarator simulation data to text format for language model training."""
    
    def __init__(self, precision: int = 4, use_scientific: bool = False):
        """
        Initialize the text converter.
        
        Args:
            precision: Number of decimal places for numerical values
            use_scientific: Whether to use scientific notation for small/large numbers
        """
        self.precision = precision
        self.use_scientific = use_scientific
        
        # Define text templates for different formats
        self.templates = {
            'structured': self._structured_template,
            'natural': self._natural_language_template,
            'json_like': self._json_like_template,
            'qa_pairs': self._qa_pairs_template,
            'code_like': self._code_like_template
        }
    
    def format_number(self, value: float) -> str:
        """Format a numerical value according to converter settings."""
        if pd.isna(value) or value is None:
            return "N/A"
        
        if self.use_scientific and (abs(value) < 1e-3 or abs(value) > 1e6):
            return f"{value:.{self.precision}e}"
        else:
            return f"{value:.{self.precision}f}"
    
    def _structured_template(self, data: Dict[str, Any]) -> str:
        """Convert data to structured text format."""
        lines = ["STELLARATOR_CONFIGURATION:"]
        
        # Input parameters
        lines.append("INPUT_PARAMETERS:")
        lines.append(f"  aspect_ratio: {self.format_number(data.get('aspect_ratio', 0))}")
        lines.append(f"  elongation: {self.format_number(data.get('elongation', 0))}")
        lines.append(f"  rotational_transform: {self.format_number(data.get('rotational_transform', 0))}")
        lines.append(f"  n_field_periods: {data.get('n_field_periods', 0)}")
        lines.append(f"  triangularity: {self.format_number(data.get('triangularity', 0))}")
        
        # Output metrics
        lines.append("OUTPUT_METRICS:")
        lines.append(f"  computed_aspect_ratio: {self.format_number(data.get('aspect_ratio_computed', 0))}")
        lines.append(f"  max_elongation: {self.format_number(data.get('max_elongation', 0))}")
        lines.append(f"  edge_rotational_transform: {self.format_number(data.get('edge_rotational_transform', 0))}")
        lines.append(f"  axis_rotational_transform: {self.format_number(data.get('axis_rotational_transform', 0))}")
        lines.append(f"  vacuum_well: {self.format_number(data.get('vacuum_well', 0))}")
        lines.append(f"  average_triangularity: {self.format_number(data.get('average_triangularity', 0))}")
        
        # Optional metrics
        if 'qi_residual' in data and pd.notna(data['qi_residual']):
            lines.append(f"  qi_residual: {self.format_number(data['qi_residual'])}")
        
        if 'flux_compression_bad_curvature' in data and pd.notna(data['flux_compression_bad_curvature']):
            lines.append(f"  flux_compression_bad_curvature: {self.format_number(data['flux_compression_bad_curvature'])}")
        
        lines.append("END_CONFIGURATION")
        return "\n".join(lines)
    
    def _natural_language_template(self, data: Dict[str, Any]) -> str:
        """Convert data to natural language description."""
        aspect_ratio = data.get('aspect_ratio', 0)
        elongation = data.get('elongation', 0)
        rotational_transform = data.get('rotational_transform', 0)
        n_field_periods = data.get('n_field_periods', 0)
        triangularity = data.get('triangularity', 0)
        
        # Computed results
        computed_aspect_ratio = data.get('aspect_ratio_computed', 0)
        max_elongation = data.get('max_elongation', 0)
        edge_rt = data.get('edge_rotational_transform', 0)
        vacuum_well = data.get('vacuum_well', 0)
        
        # Generate natural language description
        text = f"A stellarator configuration with {n_field_periods} field periods was designed with "
        text += f"an aspect ratio of {self.format_number(aspect_ratio)}, "
        text += f"elongation of {self.format_number(elongation)}, "
        text += f"rotational transform of {self.format_number(rotational_transform)}, "
        text += f"and triangularity of {self.format_number(triangularity)}. "
        
        text += f"The simulation results showed a computed aspect ratio of {self.format_number(computed_aspect_ratio)}, "
        text += f"maximum elongation of {self.format_number(max_elongation)}, "
        text += f"edge rotational transform of {self.format_number(edge_rt)}, "
        text += f"and vacuum well depth of {self.format_number(vacuum_well)}. "
        
        # Add quality assessment
        if 'qi_residual' in data and pd.notna(data['qi_residual']):
            qi_residual = data['qi_residual']
            if qi_residual < 0.1:
                text += "This configuration exhibits excellent quasi-isodynamic properties. "
            elif qi_residual < 0.5:
                text += "This configuration shows good quasi-isodynamic characteristics. "
            else:
                text += "This configuration has moderate quasi-isodynamic properties. "
        
        return text.strip()
    
    def _json_like_template(self, data: Dict[str, Any]) -> str:
        """Convert data to JSON-like structured format."""
        config = {
            "stellarator": {
                "design_parameters": {
                    "aspect_ratio": self.format_number(data.get('aspect_ratio', 0)),
                    "elongation": self.format_number(data.get('elongation', 0)),
                    "rotational_transform": self.format_number(data.get('rotational_transform', 0)),
                    "n_field_periods": data.get('n_field_periods', 0),
                    "triangularity": self.format_number(data.get('triangularity', 0))
                },
                "simulation_results": {
                    "aspect_ratio_computed": self.format_number(data.get('aspect_ratio_computed', 0)),
                    "max_elongation": self.format_number(data.get('max_elongation', 0)),
                    "edge_rotational_transform": self.format_number(data.get('edge_rotational_transform', 0)),
                    "axis_rotational_transform": self.format_number(data.get('axis_rotational_transform', 0)),
                    "vacuum_well": self.format_number(data.get('vacuum_well', 0)),
                    "average_triangularity": self.format_number(data.get('average_triangularity', 0))
                }
            }
        }
        
        # Add optional metrics
        if 'qi_residual' in data and pd.notna(data['qi_residual']):
            config["stellarator"]["simulation_results"]["qi_residual"] = self.format_number(data['qi_residual'])
        
        return json.dumps(config, indent=2)
    
    def _qa_pairs_template(self, data: Dict[str, Any]) -> str:
        """Convert data to question-answer pairs format."""
        qa_pairs = []
        
        # Design questions
        qa_pairs.append(f"Q: What is the aspect ratio of this stellarator design?")
        qa_pairs.append(f"A: {self.format_number(data.get('aspect_ratio', 0))}")
        
        qa_pairs.append(f"Q: What is the elongation?")
        qa_pairs.append(f"A: {self.format_number(data.get('elongation', 0))}")
        
        qa_pairs.append(f"Q: How many field periods does this configuration have?")
        qa_pairs.append(f"A: {data.get('n_field_periods', 0)}")
        
        # Simulation result questions
        qa_pairs.append(f"Q: What is the computed aspect ratio from the simulation?")
        qa_pairs.append(f"A: {self.format_number(data.get('aspect_ratio_computed', 0))}")
        
        qa_pairs.append(f"Q: What is the maximum elongation achieved?")
        qa_pairs.append(f"A: {self.format_number(data.get('max_elongation', 0))}")
        
        qa_pairs.append(f"Q: What is the vacuum well depth?")
        qa_pairs.append(f"A: {self.format_number(data.get('vacuum_well', 0))}")
        
        # Performance questions
        if 'qi_residual' in data and pd.notna(data['qi_residual']):
            qa_pairs.append(f"Q: What is the quasi-isodynamic residual?")
            qa_pairs.append(f"A: {self.format_number(data['qi_residual'])}")
            
            qi_residual = data['qi_residual']
            qa_pairs.append(f"Q: Is this a good quasi-isodynamic configuration?")
            if qi_residual < 0.1:
                qa_pairs.append(f"A: Yes, this is an excellent quasi-isodynamic configuration.")
            elif qi_residual < 0.5:
                qa_pairs.append(f"A: Yes, this is a good quasi-isodynamic configuration.")
            else:
                qa_pairs.append(f"A: This configuration has moderate quasi-isodynamic properties.")
        
        return "\n".join(qa_pairs)
    
    def _code_like_template(self, data: Dict[str, Any]) -> str:
        """Convert data to code-like format."""
        lines = ["# Stellarator Configuration"]
        lines.append("stellarator = StellaratorConfig(")
        lines.append(f"    aspect_ratio={self.format_number(data.get('aspect_ratio', 0))},")
        lines.append(f"    elongation={self.format_number(data.get('elongation', 0))},")
        lines.append(f"    rotational_transform={self.format_number(data.get('rotational_transform', 0))},")
        lines.append(f"    n_field_periods={data.get('n_field_periods', 0)},")
        lines.append(f"    triangularity={self.format_number(data.get('triangularity', 0))}")
        lines.append(")")
        lines.append("")
        lines.append("# Simulation Results")
        lines.append("results = simulate(stellarator)")
        lines.append(f"assert results.aspect_ratio == {self.format_number(data.get('aspect_ratio_computed', 0))}")
        lines.append(f"assert results.max_elongation == {self.format_number(data.get('max_elongation', 0))}")
        lines.append(f"assert results.vacuum_well == {self.format_number(data.get('vacuum_well', 0))}")
        
        if 'qi_residual' in data and pd.notna(data['qi_residual']):
            lines.append(f"assert results.qi_residual == {self.format_number(data['qi_residual'])}")
        
        return "\n".join(lines)
    
    def convert_dataframe(self, df: pd.DataFrame, format_type: str = 'structured') -> List[str]:
        """
        Convert a DataFrame of stellarator data to text format.
        
        Args:
            df: DataFrame containing stellarator simulation data
            format_type: Type of text format ('structured', 'natural', 'json_like', 'qa_pairs', 'code_like')
            
        Returns:
            List of text strings, one per row
        """
        if format_type not in self.templates:
            raise ValueError(f"Unknown format type: {format_type}. Available: {list(self.templates.keys())}")
        
        template_func = self.templates[format_type]
        text_data = []
        
        for _, row in df.iterrows():
            data_dict = row.to_dict()
            text = template_func(data_dict)
            text_data.append(text)
        
        return text_data
    
    def create_training_dataset(self, df: pd.DataFrame, output_dir: str, 
                              formats: List[str] = None, 
                              train_split: float = 0.8,
                              add_special_tokens: bool = True) -> Dict[str, str]:
        """
        Create training datasets in multiple text formats.
        
        Args:
            df: DataFrame containing stellarator simulation data
            output_dir: Directory to save the text datasets
            formats: List of format types to generate
            train_split: Fraction of data for training (rest for validation)
            add_special_tokens: Whether to add special tokens for language model training
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        if formats is None:
            formats = ['structured', 'natural', 'qa_pairs']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split data
        n_train = int(len(df) * train_split)
        df_train = df.iloc[:n_train]
        df_val = df.iloc[n_train:]
        
        output_files = {}
        
        for format_type in formats:
            print(f"Generating {format_type} format...")
            
            # Convert training data
            train_texts = self.convert_dataframe(df_train, format_type)
            val_texts = self.convert_dataframe(df_val, format_type)
            
            # Add special tokens if requested
            if add_special_tokens:
                train_texts = [f"<|startoftext|>{text}<|endoftext|>" for text in train_texts]
                val_texts = [f"<|startoftext|>{text}<|endoftext|>" for text in val_texts]
            
            # Save training data
            train_file = output_dir / f"train_{format_type}.txt"
            with open(train_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(train_texts))
            
            # Save validation data
            val_file = output_dir / f"val_{format_type}.txt"
            with open(val_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(val_texts))
            
            output_files[format_type] = {
                'train': str(train_file),
                'val': str(val_file)
            }
            
            print(f"  Training samples: {len(train_texts)}")
            print(f"  Validation samples: {len(val_texts)}")
            print(f"  Files: {train_file}, {val_file}")
        
        # Create a combined dataset with all formats
        if len(formats) > 1:
            print("Creating combined multi-format dataset...")
            
            all_train_texts = []
            all_val_texts = []
            
            for format_type in formats:
                train_texts = self.convert_dataframe(df_train, format_type)
                val_texts = self.convert_dataframe(df_val, format_type)
                
                if add_special_tokens:
                    train_texts = [f"<|format:{format_type}|>{text}<|endoftext|>" for text in train_texts]
                    val_texts = [f"<|format:{format_type}|>{text}<|endoftext|>" for text in val_texts]
                
                all_train_texts.extend(train_texts)
                all_val_texts.extend(val_texts)
            
            # Shuffle the combined data
            import random
            random.shuffle(all_train_texts)
            random.shuffle(all_val_texts)
            
            # Save combined data
            combined_train_file = output_dir / "train_combined.txt"
            combined_val_file = output_dir / "val_combined.txt"
            
            with open(combined_train_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(all_train_texts))
            
            with open(combined_val_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(all_val_texts))
            
            output_files['combined'] = {
                'train': str(combined_train_file),
                'val': str(combined_val_file)
            }
            
            print(f"  Combined training samples: {len(all_train_texts)}")
            print(f"  Combined validation samples: {len(all_val_texts)}")
        
        # Save metadata
        metadata = {
            'total_samples': len(df),
            'train_samples': len(df_train),
            'val_samples': len(df_val),
            'formats': formats,
            'precision': self.precision,
            'use_scientific': self.use_scientific,
            'add_special_tokens': add_special_tokens,
            'feature_columns': list(df.columns),
            'output_files': output_files
        }
        
        metadata_file = output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset creation completed!")
        print(f"Metadata saved to: {metadata_file}")
        
        return output_files
    
    def create_inference_prompt(self, input_params: Dict[str, Any], format_type: str = 'structured') -> str:
        """
        Create a prompt for inference with a trained language model.
        
        Args:
            input_params: Dictionary of input parameters
            format_type: Format type to use for the prompt
            
        Returns:
            Formatted prompt string
        """
        if format_type == 'structured':
            lines = ["STELLARATOR_CONFIGURATION:"]
            lines.append("INPUT_PARAMETERS:")
            lines.append(f"  aspect_ratio: {self.format_number(input_params.get('aspect_ratio', 0))}")
            lines.append(f"  elongation: {self.format_number(input_params.get('elongation', 0))}")
            lines.append(f"  rotational_transform: {self.format_number(input_params.get('rotational_transform', 0))}")
            lines.append(f"  n_field_periods: {input_params.get('n_field_periods', 0)}")
            lines.append(f"  triangularity: {self.format_number(input_params.get('triangularity', 0))}")
            lines.append("OUTPUT_METRICS:")
            return "\n".join(lines)
        
        elif format_type == 'natural':
            aspect_ratio = input_params.get('aspect_ratio', 0)
            elongation = input_params.get('elongation', 0)
            rotational_transform = input_params.get('rotational_transform', 0)
            n_field_periods = input_params.get('n_field_periods', 0)
            triangularity = input_params.get('triangularity', 0)
            
            prompt = f"A stellarator configuration with {n_field_periods} field periods was designed with "
            prompt += f"an aspect ratio of {self.format_number(aspect_ratio)}, "
            prompt += f"elongation of {self.format_number(elongation)}, "
            prompt += f"rotational transform of {self.format_number(rotational_transform)}, "
            prompt += f"and triangularity of {self.format_number(triangularity)}. "
            prompt += "The simulation results showed"
            return prompt
        
        elif format_type == 'qa_pairs':
            return f"Q: What are the simulation results for a stellarator with aspect ratio {self.format_number(input_params.get('aspect_ratio', 0))}, elongation {self.format_number(input_params.get('elongation', 0))}, rotational transform {self.format_number(input_params.get('rotational_transform', 0))}, {input_params.get('n_field_periods', 0)} field periods, and triangularity {self.format_number(input_params.get('triangularity', 0))}?\nA:"
        
        else:
            raise ValueError(f"Unsupported format type for inference: {format_type}")


def demo_text_conversion():
    """Demonstrate text conversion with sample data."""
    
    # Create sample data
    sample_data = [
        {
            'aspect_ratio': 4.0,
            'elongation': 1.2,
            'rotational_transform': 0.5,
            'n_field_periods': 3,
            'triangularity': 0.1,
            'aspect_ratio_computed': 4.05,
            'max_elongation': 1.25,
            'edge_rotational_transform': 0.52,
            'axis_rotational_transform': 0.48,
            'vacuum_well': 0.15,
            'average_triangularity': 0.12,
            'qi_residual': 0.08
        },
        {
            'aspect_ratio': 6.0,
            'elongation': 0.8,
            'rotational_transform': 0.3,
            'n_field_periods': 4,
            'triangularity': -0.1,
            'aspect_ratio_computed': 5.98,
            'max_elongation': 0.82,
            'edge_rotational_transform': 0.31,
            'axis_rotational_transform': 0.29,
            'vacuum_well': 0.22,
            'average_triangularity': -0.09,
            'qi_residual': 0.15
        }
    ]
    
    df = pd.DataFrame(sample_data)
    
    # Create converter
    converter = StellaratorTextConverter(precision=3)
    
    # Demonstrate different formats
    formats = ['structured', 'natural', 'qa_pairs', 'json_like', 'code_like']
    
    for format_type in formats:
        print(f"\n{'='*60}")
        print(f"FORMAT: {format_type.upper()}")
        print(f"{'='*60}")
        
        texts = converter.convert_dataframe(df, format_type)
        print(texts[0])  # Show first example
    
    # Create training dataset
    print(f"\n{'='*60}")
    print("CREATING TRAINING DATASET")
    print(f"{'='*60}")
    
    output_files = converter.create_training_dataset(
        df, 
        output_dir="./demo_text_data",
        formats=['structured', 'natural', 'qa_pairs'],
        train_split=0.5  # Small dataset, so 50/50 split
    )
    
    print("\nGenerated files:")
    for format_type, files in output_files.items():
        print(f"  {format_type}: {files}")


if __name__ == "__main__":
    demo_text_conversion()