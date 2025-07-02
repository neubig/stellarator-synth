#!/usr/bin/env python3
"""
Complete Pipeline Demo: From Synthetic Data to Language Model

This script demonstrates the complete pipeline:
1. Generate synthetic stellarator simulation data
2. Convert data to text format
3. Train a language model on the text data
4. Use the language model for physics predictions
"""

import os
import time
from pathlib import Path
import json

def run_complete_pipeline():
    """Run the complete pipeline from data generation to language model training."""
    
    print("🌟 Complete Stellarator Language Model Pipeline")
    print("=" * 70)
    
    # Configuration
    demo_config = {
        'data_generation': {
            'n_systematic_samples': 200,
            'n_lhs_samples': 300,
            'n_adaptive_samples': 100,
            'output_dir': './pipeline_data'
        },
        'text_conversion': {
            'formats': ['structured', 'natural', 'qa_pairs'],
            'output_dir': './pipeline_text_data',
            'precision': 4
        },
        'language_model': {
            'model_name': 'gpt2',
            'max_length': 256,
            'output_dir': './pipeline_language_model',
            'training_epochs': 2
        }
    }
    
    print("📋 Pipeline Configuration:")
    print(json.dumps(demo_config, indent=2))
    print()
    
    # Step 1: Generate synthetic data
    print("📊 Step 1: Generating synthetic stellarator data...")
    print("-" * 50)
    
    try:
        from synthetic_data_generator import SyntheticDataGenerator, SyntheticDataConfig
        
        config = SyntheticDataConfig(
            n_systematic_samples=demo_config['data_generation']['n_systematic_samples'],
            n_lhs_samples=demo_config['data_generation']['n_lhs_samples'],
            n_adaptive_samples=demo_config['data_generation']['n_adaptive_samples'],
            n_workers=2,
            batch_size=25,
            output_dir=demo_config['data_generation']['output_dir']
        )
        
        generator = SyntheticDataGenerator(config)
        
        start_time = time.time()
        generator.generate_all_data()
        generation_time = time.time() - start_time
        
        print(f"✅ Data generation completed in {generation_time:.1f} seconds")
        print(f"   Generated {len(generator.successful_runs)} successful simulations")
        print(f"   Failed simulations: {len(generator.failed_runs)}")
        
        if len(generator.successful_runs) < 50:
            print("⚠️  Warning: Limited data may affect language model quality")
        
    except Exception as e:
        print(f"❌ Error in data generation: {e}")
        return False
    
    print()
    
    # Step 2: Convert data to text format
    print("📝 Step 2: Converting data to text format...")
    print("-" * 50)
    
    try:
        from text_data_converter import StellaratorTextConverter
        import pandas as pd
        
        # Load the generated data
        data_file = Path(demo_config['data_generation']['output_dir']) / "successful_runs_final.csv"
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            return False
        
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} successful simulations")
        
        # Convert to text
        converter = StellaratorTextConverter(precision=demo_config['text_conversion']['precision'])
        
        output_files = converter.create_training_dataset(
            df,
            output_dir=demo_config['text_conversion']['output_dir'],
            formats=demo_config['text_conversion']['formats'],
            train_split=0.8,
            add_special_tokens=True
        )
        
        print("✅ Text conversion completed")
        print("   Generated formats:", list(output_files.keys()))
        
        # Show example text
        example_file = Path(demo_config['text_conversion']['output_dir']) / "train_structured.txt"
        if example_file.exists():
            with open(example_file, 'r') as f:
                example_text = f.read()[:500]  # First 500 characters
            print("\n📄 Example text format:")
            print("-" * 30)
            print(example_text + "...")
        
    except Exception as e:
        print(f"❌ Error in text conversion: {e}")
        return False
    
    print()
    
    # Step 3: Train language model
    print("🤖 Step 3: Training language model...")
    print("-" * 50)
    
    try:
        from language_model_trainer import StellaratorLanguageModel
        
        # Check if transformers is available
        try:
            import transformers
            print(f"Using transformers version: {transformers.__version__}")
        except ImportError:
            print("❌ Transformers library not available. Install with:")
            print("   pip install transformers datasets torch")
            return False
        
        # Create and setup model
        model = StellaratorLanguageModel(
            model_name=demo_config['language_model']['model_name'],
            max_length=demo_config['language_model']['max_length']
        )
        
        # Load text data
        train_texts, val_texts = model.load_text_data(
            demo_config['text_conversion']['output_dir'], 
            'structured'
        )
        
        print(f"Training on {len(train_texts)} examples, validating on {len(val_texts)} examples")
        
        # Training arguments for demo (reduced for speed)
        training_args = {
            'num_train_epochs': demo_config['language_model']['training_epochs'],
            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'warmup_steps': 20,
            'logging_steps': 20,
            'save_steps': 200,
            'eval_steps': 200,
            'learning_rate': 5e-5,
            'fp16': False,  # Disable for compatibility
        }
        
        # Train model
        start_time = time.time()
        model.train_model(
            train_texts,
            val_texts,
            output_dir=demo_config['language_model']['output_dir'],
            training_args=training_args
        )
        training_time = time.time() - start_time
        
        print(f"✅ Language model training completed in {training_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error in language model training: {e}")
        print("This might be due to insufficient computational resources or missing dependencies.")
        return False
    
    print()
    
    # Step 4: Test the trained model
    print("🔮 Step 4: Testing the trained language model...")
    print("-" * 50)
    
    try:
        # Test different stellarator configurations
        test_configurations = [
            {
                'name': 'Compact Stellarator',
                'params': {
                    'aspect_ratio': 3.5,
                    'elongation': 1.0,
                    'rotational_transform': 0.6,
                    'n_field_periods': 3,
                    'triangularity': 0.1
                }
            },
            {
                'name': 'Large Stellarator',
                'params': {
                    'aspect_ratio': 7.0,
                    'elongation': 0.8,
                    'rotational_transform': 0.3,
                    'n_field_periods': 5,
                    'triangularity': -0.1
                }
            },
            {
                'name': 'High Elongation Design',
                'params': {
                    'aspect_ratio': 5.0,
                    'elongation': 1.8,
                    'rotational_transform': 0.4,
                    'n_field_periods': 4,
                    'triangularity': 0.2
                }
            }
        ]
        
        for config in test_configurations:
            print(f"\n🔬 Testing: {config['name']}")
            print("Input parameters:")
            for key, value in config['params'].items():
                print(f"  {key}: {value}")
            
            # Get prediction
            prediction = model.predict_stellarator_metrics(config['params'], 'structured')
            
            print("Predicted metrics:")
            if prediction:
                for key, value in prediction.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("  No metrics extracted from prediction")
        
        print("\n✅ Model testing completed!")
        
    except Exception as e:
        print(f"❌ Error in model testing: {e}")
        return False
    
    print()
    
    # Step 5: Generate summary and next steps
    print("📋 Step 5: Pipeline Summary")
    print("-" * 50)
    
    summary = {
        'data_generation': {
            'successful_simulations': len(generator.successful_runs),
            'generation_time_seconds': generation_time,
            'output_directory': demo_config['data_generation']['output_dir']
        },
        'text_conversion': {
            'formats_generated': demo_config['text_conversion']['formats'],
            'output_directory': demo_config['text_conversion']['output_dir']
        },
        'language_model': {
            'model_type': demo_config['language_model']['model_name'],
            'training_time_seconds': training_time,
            'output_directory': demo_config['language_model']['output_dir']
        }
    }
    
    print("Pipeline completed successfully! 🎉")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    
    print("\n🚀 Next Steps:")
    print("1. Scale up data generation for better model performance")
    print("2. Experiment with different language model architectures")
    print("3. Fine-tune hyperparameters for your specific use case")
    print("4. Integrate the model into stellarator optimization workflows")
    print("5. Validate predictions against real experimental data")
    
    print("\n📁 Generated Files:")
    print(f"• Synthetic data: {demo_config['data_generation']['output_dir']}/")
    print(f"• Text datasets: {demo_config['text_conversion']['output_dir']}/")
    print(f"• Trained model: {demo_config['language_model']['output_dir']}/")
    
    # Save pipeline summary
    summary_file = Path("./pipeline_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"• Pipeline summary: {summary_file}")
    
    return True


def main():
    """Main function to run the complete pipeline."""
    
    print("🌟 Stellarator Language Model Pipeline Demo")
    print("This demo will:")
    print("1. Generate synthetic stellarator simulation data")
    print("2. Convert the data to text format suitable for language models")
    print("3. Train a language model on the text data")
    print("4. Test the model's ability to predict physics metrics")
    print()
    
    # Check dependencies
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   {dep}")
        print("\nInstall with: pip install torch transformers datasets")
        return
    
    # Run pipeline
    success = run_complete_pipeline()
    
    if success:
        print("\n🎉 Complete pipeline demo finished successfully!")
        print("You now have a working language model for stellarator physics!")
    else:
        print("\n❌ Pipeline demo encountered errors.")
        print("Check the error messages above for troubleshooting.")


if __name__ == "__main__":
    main()