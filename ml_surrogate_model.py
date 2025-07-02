#!/usr/bin/env python3
"""
Machine Learning Surrogate Model for Stellarator Simulation

This script demonstrates how to train ML models on the synthetic data
to create fast surrogate models that mimic the VMEC++ simulator.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Optional: Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class StellaratorSurrogateModel:
    """Machine learning surrogate model for stellarator simulation."""
    
    def __init__(self, data_path: str, model_type: str = "random_forest"):
        """
        Initialize the surrogate model.
        
        Args:
            data_path: Path to the synthetic data directory
            model_type: Type of ML model to use
        """
        self.data_path = Path(data_path)
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.target_names = []
        self.training_stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load the synthetic training data."""
        # Look for the final successful runs file
        data_files = list(self.data_path.glob("successful_runs_final.csv"))
        if not data_files:
            # Fall back to any successful runs file
            data_files = list(self.data_path.glob("successful_runs*.csv"))
        
        if not data_files:
            raise FileNotFoundError(f"No successful runs data found in {self.data_path}")
        
        # Load the most recent file
        data_file = max(data_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading data from: {data_file}")
        
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} successful simulations")
        
        return df
    
    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target matrix from the data."""
        
        # Define input features (parameters that define the stellarator)
        feature_columns = [
            'aspect_ratio',
            'elongation', 
            'rotational_transform',
            'n_field_periods',
            'triangularity'
        ]
        
        # Define target variables (simulation outputs we want to predict)
        target_columns = [
            'aspect_ratio_computed',
            'max_elongation',
            'edge_rotational_transform',
            'axis_rotational_transform',
            'vacuum_well',
            'average_triangularity',
            'axis_magnetic_mirror_ratio',
            'edge_magnetic_mirror_ratio',
            'min_normalized_magnetic_gradient_scale_length'
        ]
        
        # Add optional targets if available
        optional_targets = ['qi_residual', 'flux_compression_bad_curvature']
        for target in optional_targets:
            if target in df.columns and not df[target].isna().all():
                target_columns.append(target)
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in df.columns]
        available_targets = [col for col in target_columns if col in df.columns]
        
        print(f"Using {len(available_features)} features: {available_features}")
        print(f"Predicting {len(available_targets)} targets: {available_targets}")
        
        # Extract feature and target matrices
        X = df[available_features].values
        y = df[available_targets].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        self.feature_names = available_features
        self.target_names = available_targets
        
        return X, y
    
    def create_model(self, n_targets: int):
        """Create the ML model based on the specified type."""
        
        if self.model_type == "random_forest":
            base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == "gradient_boosting":
            base_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        elif self.model_type == "neural_network":
            base_model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=1000,
                random_state=42
            )
        
        elif self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            base_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            base_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Wrap in MultiOutputRegressor for multiple targets
        if n_targets > 1:
            return MultiOutputRegressor(base_model, n_jobs=-1)
        else:
            return base_model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Train the surrogate model."""
        
        print(f"Training {self.model_type} model...")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target matrix shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        self.scalers['features'] = RobustScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        # Scale targets (optional, but can help with neural networks)
        if self.model_type == "neural_network":
            self.scalers['targets'] = RobustScaler()
            y_train_scaled = self.scalers['targets'].fit_transform(y_train)
            y_test_scaled = self.scalers['targets'].transform(y_test)
        else:
            y_train_scaled = y_train
            y_test_scaled = y_test
        
        # Create and train model
        self.models['main'] = self.create_model(y.shape[1])
        self.models['main'].fit(X_train_scaled, y_train_scaled)
        
        # Make predictions
        y_pred_train = self.models['main'].predict(X_train_scaled)
        y_pred_test = self.models['main'].predict(X_test_scaled)
        
        # Inverse transform if needed
        if self.model_type == "neural_network" and 'targets' in self.scalers:
            y_pred_train = self.scalers['targets'].inverse_transform(y_pred_train)
            y_pred_test = self.scalers['targets'].inverse_transform(y_pred_test)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_pred_train)
        test_metrics = self.calculate_metrics(y_test, y_pred_test)
        
        self.training_stats = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_type': self.model_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print("\nTraining completed!")
        print(f"Training R²: {train_metrics['r2_mean']:.4f} ± {train_metrics['r2_std']:.4f}")
        print(f"Test R²: {test_metrics['r2_mean']:.4f} ± {test_metrics['r2_std']:.4f}")
        print(f"Test RMSE: {test_metrics['rmse_mean']:.4f} ± {test_metrics['rmse_std']:.4f}")
        
        return X_test, y_test, y_pred_test
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics for each target."""
        
        metrics = {
            'rmse_per_target': [],
            'mae_per_target': [],
            'r2_per_target': []
        }
        
        for i in range(y_true.shape[1]):
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            
            metrics['rmse_per_target'].append(rmse)
            metrics['mae_per_target'].append(mae)
            metrics['r2_per_target'].append(r2)
        
        # Overall metrics
        metrics['rmse_mean'] = np.mean(metrics['rmse_per_target'])
        metrics['rmse_std'] = np.std(metrics['rmse_per_target'])
        metrics['mae_mean'] = np.mean(metrics['mae_per_target'])
        metrics['mae_std'] = np.std(metrics['mae_per_target'])
        metrics['r2_mean'] = np.mean(metrics['r2_per_target'])
        metrics['r2_std'] = np.std(metrics['r2_per_target'])
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if 'main' not in self.models:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Scale features
        X_scaled = self.scalers['features'].transform(X)
        
        # Make prediction
        y_pred = self.models['main'].predict(X_scaled)
        
        # Inverse transform if needed
        if self.model_type == "neural_network" and 'targets' in self.scalers:
            y_pred = self.scalers['targets'].inverse_transform(y_pred)
        
        return y_pred
    
    def save_model(self, save_path: str):
        """Save the trained model and scalers."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = save_path / "surrogate_model.pkl"
        joblib.dump(self.models['main'], model_file)
        
        # Save scalers
        scalers_file = save_path / "scalers.pkl"
        joblib.dump(self.scalers, scalers_file)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_type': self.model_type,
            'training_stats': self.training_stats
        }
        metadata_file = save_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to: {save_path}")
    
    def load_model(self, load_path: str):
        """Load a previously trained model."""
        load_path = Path(load_path)
        
        # Load model
        model_file = load_path / "surrogate_model.pkl"
        self.models['main'] = joblib.load(model_file)
        
        # Load scalers
        scalers_file = load_path / "scalers.pkl"
        self.scalers = joblib.load(scalers_file)
        
        # Load metadata
        metadata_file = load_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.target_names = metadata['target_names']
        self.model_type = metadata['model_type']
        self.training_stats = metadata['training_stats']
        
        print(f"Model loaded from: {load_path}")
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
        """Plot prediction vs true values for each target."""
        
        n_targets = len(self.target_names)
        n_cols = min(3, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_targets == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, target_name in enumerate(self.target_names):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break
                
            # Scatter plot
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Labels and title
            ax.set_xlabel(f'True {target_name}')
            ax.set_ylabel(f'Predicted {target_name}')
            
            # R² score
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            ax.set_title(f'{target_name}\nR² = {r2:.3f}')
            
            # Equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
        
        # Hide unused subplots
        for i in range(n_targets, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction plots saved to: {save_path}")
        
        plt.show()
    
    def benchmark_speed(self, n_samples: int = 1000) -> Dict[str, float]:
        """Benchmark the speed of the surrogate model."""
        
        if 'main' not in self.models:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Generate random test data
        X_test = np.random.randn(n_samples, len(self.feature_names))
        
        # Time predictions
        import time
        start_time = time.time()
        y_pred = self.predict(X_test)
        end_time = time.time()
        
        total_time = end_time - start_time
        time_per_sample = total_time / n_samples
        
        results = {
            'total_time_seconds': total_time,
            'time_per_sample_ms': time_per_sample * 1000,
            'samples_per_second': n_samples / total_time,
            'n_samples': n_samples
        }
        
        print(f"\nSpeed Benchmark Results:")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Time per sample: {time_per_sample*1000:.4f} ms")
        print(f"Samples per second: {results['samples_per_second']:.1f}")
        
        return results


def compare_models(data_path: str, model_types: List[str] = None):
    """Compare different model types on the same dataset."""
    
    if model_types is None:
        model_types = ["random_forest", "gradient_boosting", "neural_network"]
        if XGBOOST_AVAILABLE:
            model_types.append("xgboost")
        if LIGHTGBM_AVAILABLE:
            model_types.append("lightgbm")
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type} model...")
        print(f"{'='*50}")
        
        try:
            # Create and train model
            surrogate = StellaratorSurrogateModel(data_path, model_type)
            df = surrogate.load_data()
            X, y = surrogate.prepare_features_and_targets(df)
            X_test, y_test, y_pred = surrogate.train_model(X, y)
            
            # Benchmark speed
            speed_results = surrogate.benchmark_speed()
            
            # Store results
            results[model_type] = {
                'test_metrics': surrogate.training_stats['test_metrics'],
                'speed_results': speed_results,
                'model': surrogate
            }
            
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    # Print comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}")
    
    print(f"{'Model':<15} {'R²':<8} {'RMSE':<8} {'Speed (ms)':<12}")
    print("-" * 50)
    
    for model_type, result in results.items():
        if 'error' in result:
            print(f"{model_type:<15} {'ERROR':<8} {'ERROR':<8} {'ERROR':<12}")
        else:
            r2 = result['test_metrics']['r2_mean']
            rmse = result['test_metrics']['rmse_mean']
            speed = result['speed_results']['time_per_sample_ms']
            print(f"{model_type:<15} {r2:<8.3f} {rmse:<8.3f} {speed:<12.2f}")
    
    return results


def main():
    """Main function to demonstrate the surrogate model training."""
    
    # Configuration
    data_path = "./demo_synthetic_data"  # Path to synthetic data
    model_type = "random_forest"  # Model type to use
    
    print("Stellarator Surrogate Model Training Demo")
    print("=" * 50)
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"Data directory not found: {data_path}")
        print("Please run synthetic_data_generator.py first to generate training data.")
        return
    
    # Create and train surrogate model
    surrogate = StellaratorSurrogateModel(data_path, model_type)
    
    try:
        # Load data
        df = surrogate.load_data()
        
        # Prepare features and targets
        X, y = surrogate.prepare_features_and_targets(df)
        
        # Train model
        X_test, y_test, y_pred = surrogate.train_model(X, y)
        
        # Plot predictions
        surrogate.plot_predictions(y_test, y_pred, "prediction_plots.png")
        
        # Benchmark speed
        surrogate.benchmark_speed()
        
        # Save model
        surrogate.save_model("./trained_surrogate_model")
        
        # Demonstrate usage
        print("\nDemonstrating surrogate model usage:")
        print("-" * 40)
        
        # Create example input
        example_input = np.array([[
            4.0,    # aspect_ratio
            1.2,    # elongation
            0.5,    # rotational_transform
            3,      # n_field_periods
            0.1     # triangularity
        ]])
        
        # Make prediction
        prediction = surrogate.predict(example_input)
        
        print("Input parameters:")
        for i, feature in enumerate(surrogate.feature_names):
            print(f"  {feature}: {example_input[0, i]}")
        
        print("\nPredicted outputs:")
        for i, target in enumerate(surrogate.target_names):
            print(f"  {target}: {prediction[0, i]:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have generated synthetic data first.")


if __name__ == "__main__":
    main()