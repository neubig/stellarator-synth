"""Tests for the example data generator."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.example_data import (
    generate_boundary,
    generate_equilibrium,
    extract_input_features,
    extract_output_features,
    generate_example_dataset,
)


class TestExampleData(unittest.TestCase):
    """Test the example data generator functions."""

    def test_generate_boundary(self):
        """Test the boundary generation function."""
        n_field_periods = 5
        max_poloidal_mode = 3
        max_toroidal_mode = 3
        
        boundary = generate_boundary(
            n_field_periods=n_field_periods,
            max_poloidal_mode=max_poloidal_mode,
            max_toroidal_mode=max_toroidal_mode
        )
        
        self.assertEqual(boundary["n_field_periods"], n_field_periods)
        self.assertTrue(boundary["is_stellarator_symmetric"])
        self.assertEqual(len(boundary["rbc"]), max_poloidal_mode + 1)
        self.assertEqual(len(boundary["rbc"][0]), max_toroidal_mode + 1)
        self.assertEqual(len(boundary["zbs"]), max_poloidal_mode + 1)
        self.assertEqual(len(boundary["zbs"][0]), max_toroidal_mode + 1)
        self.assertEqual(boundary["rbc"][0][0], 1.0)  # Major radius

    def test_generate_equilibrium(self):
        """Test the equilibrium generation function."""
        boundary = generate_boundary(n_field_periods=5)
        equilibrium = generate_equilibrium(boundary)
        
        self.assertEqual(equilibrium["n_field_periods"], boundary["n_field_periods"])
        self.assertGreater(equilibrium["aspect"], 0)
        self.assertGreater(equilibrium["volume"], 0)
        self.assertGreater(equilibrium["b0"], 0)
        self.assertEqual(len(equilibrium["iota_full"]), 10)

    def test_extract_input_features(self):
        """Test the input feature extraction function."""
        boundary = generate_boundary(n_field_periods=5)
        features = extract_input_features(boundary)
        
        self.assertEqual(features["n_field_periods"], boundary["n_field_periods"])
        self.assertEqual(features["is_stellarator_symmetric"], int(boundary["is_stellarator_symmetric"]))
        self.assertEqual(features["rbc_0_0"], boundary["rbc"][0][0])

    def test_extract_output_features(self):
        """Test the output feature extraction function."""
        boundary = generate_boundary(n_field_periods=5)
        equilibrium = generate_equilibrium(boundary)
        features = extract_output_features(equilibrium)
        
        self.assertEqual(features["aspect_ratio"], equilibrium["aspect"])
        self.assertEqual(features["volume"], equilibrium["volume"])
        self.assertEqual(features["b0"], equilibrium["b0"])

    def test_generate_example_dataset(self):
        """Test the dataset generation function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            n_samples = 3
            input_df, output_df = generate_example_dataset(
                n_samples=n_samples,
                output_dir=temp_dir
            )
            
            # Check dataframes
            self.assertEqual(len(input_df), n_samples)
            self.assertEqual(len(output_df), n_samples)
            
            # Check files
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "inputs.csv")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "outputs.csv")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "metadata.json")))
            
            for i in range(n_samples):
                self.assertTrue(os.path.exists(os.path.join(temp_dir, f"boundary_{i}.json")))
                self.assertTrue(os.path.exists(os.path.join(temp_dir, f"equilibrium_{i}.json")))


if __name__ == "__main__":
    unittest.main()