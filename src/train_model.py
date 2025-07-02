#!/usr/bin/env python3
"""
Train a surrogate model to mimic the stellarator simulator.
This script loads the processed dataset and trains a neural network model.
"""

import os
import numpy as np
import pandas as pd
import argparse
import logging
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential, load_model
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
except ImportError as e:
    logger.error(f"Error importing TensorFlow: {e}")
    logger.error("Please install TensorFlow: pip install tensorflow")
    raise


def create_surrogate_model(input_dim, output_dim, layer_sizes=[256, 128, 64]):
    """
    Create a neural network surrogate model for the stellarator simulator.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output features
        layer_sizes: List of hidden layer sizes
        
    Returns:
        model: Compiled TensorFlow model
    """
    logger.info(f"Creating model with input_dim={input_dim}, output_dim={output_dim}")
    logger.info(f"Hidden layers: {layer_sizes}")
    
    # Input layer
    inputs = Input(shape=(input_dim,))
    x = inputs
    
    # Hidden layers
    for i, units in enumerate(layer_sizes):
        x = Dense(units, activation='relu', name=f'dense_{i}')(x)
        x = BatchNormalization(name=f'bn_{i}')(x)
        x = Dropout(0.2, name=f'dropout_{i}')(x)
    
    # Output layer
    outputs = Dense(output_dim, activation='linear', name='output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    logger.info(f"Model created with {model.count_params()} parameters")
    model.summary(print_fn=logger.info)
    
    return model


def train_surrogate_model(model, X_train, y_train, X_val, y_val, output_dir="model", 
                         batch_size=32, epochs=200, patience=20):
    """
    Train the surrogate model.
    
    Args:
        model: TensorFlow model to train
        X_train: Training input features
        y_train: Training target values
        X_val: Validation input features
        y_val: Validation target values
        output_dir: Directory to save model and results
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Patience for early stopping
        
    Returns:
        tuple: (model, history) Trained model and training history
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Training model with {X_train.shape[0]} samples")
    logger.info(f"Validation set size: {X_val.shape[0]} samples")
    logger.info(f"Batch size: {batch_size}, Max epochs: {epochs}, Patience: {patience}")
    
    # Create TensorBoard log directory
    log_dir = os.path.join(output_dir, "logs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'model_checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 4,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Train the model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Final training loss: {history.history['loss'][-1]:.6f}")
    logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    
    # Save the final model
    model.save(os.path.join(output_dir, 'surrogate_model.h5'))
    
    # Save training history
    pd.DataFrame(history.history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Save training metadata
    metadata = {
        "training_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "input_dim": X_train.shape[1],
        "output_dim": y_train.shape[1],
        "batch_size": batch_size,
        "epochs_run": len(history.history['loss']),
        "max_epochs": epochs,
        "patience": patience,
        "training_time_seconds": training_time,
        "final_training_loss": float(history.history['loss'][-1]),
        "final_validation_loss": float(history.history['val_loss'][-1]),
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    return model, history


def evaluate_surrogate_model(model, X_test, y_test, scaler_y, target_names, output_dir="model"):
    """
    Evaluate the surrogate model on the test set.
    
    Args:
        model: Trained TensorFlow model
        X_test: Test input features
        y_test: Test target values
        scaler_y: Scaler used to normalize target values
        target_names: Names of target features
        output_dir: Directory to save evaluation results
        
    Returns:
        DataFrame: Evaluation metrics for each target
    """
    logger.info(f"Evaluating model on {X_test.shape[0]} test samples")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform to get original scale
    y_test_orig = scaler_y.inverse_transform(y_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig, multioutput='raw_values')
    mae = mean_absolute_error(y_test_orig, y_pred_orig, multioutput='raw_values')
    r2 = r2_score(y_test_orig, y_pred_orig, multioutput='raw_values')
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Add small epsilon to avoid division by zero
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / (np.abs(y_test_orig) + 1e-10)) * 100, axis=0)
    
    # Create a results dataframe
    results = pd.DataFrame({
        'Target': target_names,
        'MSE': mse,
        'MAE': mae,
        'MAPE (%)': mape,
        'R²': r2
    })
    
    # Save results
    results.to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)
    
    # Log summary
    logger.info(f"Overall MSE: {np.mean(mse):.6f}")
    logger.info(f"Overall MAE: {np.mean(mae):.6f}")
    logger.info(f"Overall R²: {np.mean(r2):.6f}")
    
    # Plot predictions vs actual for a few key metrics
    # Try to identify key metrics, or use the first few if not found
    key_metrics = ['aspect_ratio', 'max_elongation', 'vacuum_well']
    key_indices = []
    
    for metric in key_metrics:
        for i, name in enumerate(target_names):
            if metric in name:
                key_indices.append(i)
                break
    
    # If no key metrics found, use the first 3
    if not key_indices and len(target_names) > 0:
        key_indices = list(range(min(3, len(target_names))))
    
    if key_indices:
        fig, axes = plt.subplots(1, len(key_indices), figsize=(15, 5))
        
        for i, idx in enumerate(key_indices):
            ax = axes[i] if len(key_indices) > 1 else axes
            ax.scatter(y_test_orig[:, idx], y_pred_orig[:, idx], alpha=0.5)
            ax.plot([y_test_orig[:, idx].min(), y_test_orig[:, idx].max()], 
                    [y_test_orig[:, idx].min(), y_test_orig[:, idx].max()], 
                    'r--')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(target_names[idx])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'))
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stellarator surrogate model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with processed data")
    parser.add_argument("--output_dir", type=str, default="model", help="Directory to save model and results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--layer_sizes", type=str, default="256,128,64", help="Hidden layer sizes (comma-separated)")
    
    args = parser.parse_args()
    
    # Parse layer sizes
    layer_sizes = [int(size) for size in args.layer_sizes.split(",")]
    
    # Load the processed data
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    X_val = np.load(os.path.join(args.data_dir, "X_val.npy"))
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    y_val = np.load(os.path.join(args.data_dir, "y_val.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))
    
    # Load the scaler and target names
    scaler_y = joblib.load(os.path.join(args.data_dir, "scaler_y.pkl"))
    target_names = pd.read_csv(os.path.join(args.data_dir, "target_names.csv"))["target"].tolist()
    
    # Create and train the model
    model = create_surrogate_model(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        layer_sizes=layer_sizes
    )
    
    model, history = train_surrogate_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience
    )
    
    # Evaluate the model
    results = evaluate_surrogate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler_y=scaler_y,
        target_names=target_names,
        output_dir=args.output_dir
    )