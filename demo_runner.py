#!/usr/bin/env python3
"""
Demo runner for stellarator synthetic data generation.

This script demonstrates the complete pipeline from data generation to ML model training.
"""

import os
import sys
import time
from pathlib import Path

def run_demo():
    """Run the complete demo pipeline."""
    
    print("🌟 Stellarator Synthetic Data Generation Demo")
    print("=" * 60)
    
    # Check if we're in demo mode (without actual simulators)
    demo_mode = True
    try:
        import constellaration
        demo_mode = False
        print("✅ ConStellaration framework detected - running with real simulators")
    except ImportError:
        print("⚠️  ConStellaration not available - running in demo mode")
        print("   (This will generate mock data for demonstration purposes)")
    
    print()
    
    # Step 1: Generate synthetic data
    print("📊 Step 1: Generating synthetic data...")
    print("-" * 40)
    
    from synthetic_data_generator import SyntheticDataGenerator, SyntheticDataConfig
    
    # Use smaller numbers for demo
    config = SyntheticDataConfig(
        n_systematic_samples=100,
        n_lhs_samples=200,
        n_adaptive_samples=100,
        n_workers=2,
        batch_size=25,
        output_dir="./demo_output"
    )
    
    generator = SyntheticDataGenerator(config)
    
    start_time = time.time()
    generator.generate_all_data()
    generation_time = time.time() - start_time
    
    print(f"✅ Data generation completed in {generation_time:.1f} seconds")
    print(f"   Generated {len(generator.successful_runs)} successful simulations")
    print(f"   Failed simulations: {len(generator.failed_runs)}")
    print()
    
    # Step 2: Train ML models
    if len(generator.successful_runs) > 10:  # Need minimum data for training
        print("🤖 Step 2: Training ML surrogate models...")
        print("-" * 40)
        
        from ml_surrogate_model import StellaratorSurrogateModel
        
        # Train a simple model
        surrogate = StellaratorSurrogateModel("./demo_output", "random_forest")
        
        try:
            df = surrogate.load_data()
            X, y = surrogate.prepare_features_and_targets(df)
            
            if len(X) > 20:  # Need minimum samples for train/test split
                X_test, y_test, y_pred = surrogate.train_model(X, y, test_size=0.3)
                
                # Save the model
                surrogate.save_model("./demo_model")
                
                # Benchmark speed
                speed_results = surrogate.benchmark_speed(n_samples=100)
                
                print("✅ ML model training completed")
                print(f"   Test R²: {surrogate.training_stats['test_metrics']['r2_mean']:.3f}")
                print(f"   Prediction speed: {speed_results['time_per_sample_ms']:.2f} ms per sample")
                print()
                
                # Step 3: Demonstrate usage
                print("🔮 Step 3: Demonstrating model usage...")
                print("-" * 40)
                
                import numpy as np
                
                # Example stellarator parameters
                example_params = np.array([[
                    4.0,    # aspect_ratio
                    1.2,    # elongation
                    0.5,    # rotational_transform
                    3,      # n_field_periods
                    0.1     # triangularity
                ]])
                
                prediction = surrogate.predict(example_params)
                
                print("Example prediction:")
                print("Input parameters:")
                for i, feature in enumerate(surrogate.feature_names):
                    print(f"  {feature}: {example_params[0, i]}")
                
                print("\nPredicted outputs:")
                for i, target in enumerate(surrogate.target_names):
                    print(f"  {target}: {prediction[0, i]:.4f}")
                
            else:
                print("⚠️  Not enough data for ML training (need >20 samples)")
        
        except Exception as e:
            print(f"❌ Error in ML training: {e}")
    
    else:
        print("⚠️  Not enough successful simulations for ML training")
    
    print()
    print("🎉 Demo completed!")
    print("=" * 60)
    
    # Summary
    print("📋 Summary:")
    print(f"   • Data generation time: {generation_time:.1f} seconds")
    print(f"   • Successful simulations: {len(generator.successful_runs)}")
    print(f"   • Output directory: ./demo_output")
    if os.path.exists("./demo_model"):
        print(f"   • Trained model saved to: ./demo_model")
    
    if demo_mode:
        print("\n💡 To run with real physics simulators:")
        print("   pip install constellaration")
        print("   sudo apt-get install build-essential cmake libnetcdf-dev")
    
    print("\n📚 Next steps:")
    print("   • Explore the generated data in ./demo_output/")
    print("   • Modify parameters in synthetic_data_generator.py")
    print("   • Try different ML models in ml_surrogate_model.py")
    print("   • Scale up for production use")


if __name__ == "__main__":
    run_demo()