#!/usr/bin/env python3
"""
Process the raw stellarator dataset into a format suitable for machine learning.
This script handles data cleaning, normalization, and train/validation/test splitting.
"""

import os
import numpy as np
import pandas as pd
import argparse
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_ml_dataset(input_csv, output_csv, output_dir="ml_dataset", test_size=0.2, val_size=0.15):
    """
    Process the raw data into a format suitable for ML training.
    
    Args:
        input_csv: Path to the input features CSV
        output_csv: Path to the output features CSV
        output_dir: Directory to save the processed data
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        
    Returns:
        dict: Dictionary containing dataset statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Processing dataset from {input_csv} and {output_csv}")
    
    # Load the data
    X = pd.read_csv(input_csv)
    y = pd.read_csv(output_csv)
    
    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Log dataset statistics
    logger.info(f"Dataset size: {len(X)} samples")
    logger.info(f"Input features: {X.shape[1]}")
    logger.info(f"Output features: {y.shape[1]}")
    
    # Split into train/validation/test sets
    # First split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Then split the remaining data into train and validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Normalize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Save the processed datasets
    np.save(os.path.join(output_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val_scaled)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train_scaled)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val_scaled)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test_scaled)
    
    # Save the original data for reference
    X_train.to_csv(os.path.join(output_dir, "X_train_original.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "X_val_original.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test_original.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train_original.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val_original.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test_original.csv"), index=False)
    
    # Save the scalers for later use
    joblib.dump(scaler_X, os.path.join(output_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(output_dir, "scaler_y.pkl"))
    
    # Save column names for reference
    pd.DataFrame({"feature": X.columns}).to_csv(os.path.join(output_dir, "feature_names.csv"), index=False)
    pd.DataFrame({"target": y.columns}).to_csv(os.path.join(output_dir, "target_names.csv"), index=False)
    
    # Calculate and save feature statistics
    feature_stats = X.describe().transpose()
    feature_stats.to_csv(os.path.join(output_dir, "feature_statistics.csv"))
    
    target_stats = y.describe().transpose()
    target_stats.to_csv(os.path.join(output_dir, "target_statistics.csv"))
    
    # Save metadata
    metadata = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_targets": y.shape[1],
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "processing_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(os.path.join(output_dir, "processing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Processed dataset saved to {output_dir}")
    
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process stellarator dataset for ML")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input features CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output features CSV")
    parser.add_argument("--output_dir", type=str, default="ml_dataset", help="Directory to save processed data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--val_size", type=float, default=0.15, help="Fraction of data for validation")
    
    args = parser.parse_args()
    
    prepare_ml_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size
    )