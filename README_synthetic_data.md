# Stellarator Synthetic Data Generation

This repository contains tools and documentation for generating large-scale synthetic training data that can be used to train machine learning models to mimic stellarator physics simulators.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/neubig/stellarator-synth.git
cd stellarator-synth

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (demo mode)
python synthetic_data_generator.py

# Train ML surrogate model
python ml_surrogate_model.py
```

## Overview

The stellarator simulators mentioned in the Hugging Face blog post are **open source** and can indeed be used to generate synthetic training data for machine learning models:

### Open Source Simulators ✅

1. **VMEC++** - MIT License
   - Modern C++ reimplementation of the VMEC physics simulator
   - Repository: https://github.com/proximafusion/vmecpp
   - Documentation: https://proximafusion.github.io/vmecpp/

2. **ConStellaration Framework** - Open Source
   - Optimization and analysis tools for stellarator design
   - Repository: https://github.com/proximafusion/constellaration
   - Dataset: https://huggingface.co/datasets/proxima-fusion/constellaration

3. **Existing Dataset**
   - 150,000+ QI equilibria already available
   - Can be extended with additional synthetic data

## Files in This Repository

### 1. `synthetic_data_generation_plan.md`
Comprehensive plan for generating synthetic training data, including:
- Systematic parameter space exploration
- Latin Hypercube Sampling (LHS)
- Adaptive sampling techniques
- Multi-fidelity data generation
- Resource requirements and timelines

### 2. `synthetic_data_generator.py`
Python script that implements the data generation pipeline:
- Systematic grid sampling across parameter space
- LHS for space-filling designs
- Adaptive sampling for underrepresented regions
- Parallel processing for efficiency
- Robust error handling and data validation

### 3. `ml_surrogate_model.py`
Machine learning framework for training surrogate models:
- Multiple ML algorithms (Random Forest, XGBoost, Neural Networks)
- Multi-output regression for predicting multiple physics quantities
- Model comparison and benchmarking
- Speed optimization for real-time inference

## Quick Start

### Prerequisites

```bash
# Install ConStellaration framework
pip install constellaration

# Install ML dependencies
pip install scikit-learn pandas numpy matplotlib seaborn
pip install xgboost lightgbm  # Optional: for advanced models

# System dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake libnetcdf-dev
```

### Generate Synthetic Data

```bash
# Run the data generation script
python synthetic_data_generator.py

# This will create a directory with synthetic data:
# ./demo_synthetic_data/
#   ├── successful_runs_final.csv
#   ├── failed_runs_final.csv
#   ├── generation_summary.json
#   └── generation.log
```

### Train ML Surrogate Model

```bash
# Train a surrogate model on the generated data
python ml_surrogate_model.py

# This will:
# 1. Load the synthetic data
# 2. Train multiple ML models
# 3. Evaluate performance
# 4. Save the best model
# 5. Generate prediction plots
```

## Data Generation Strategy

### Phase 1: Systematic Exploration
- **Grid Sampling**: Systematic exploration of key parameters
- **Parameters**: aspect ratio, elongation, rotational transform, field periods, triangularity
- **Target**: 10,000-25,000 configurations

### Phase 2: Space-Filling Sampling
- **Latin Hypercube Sampling**: Efficient coverage of parameter space
- **Adaptive Sampling**: Focus on regions with high model uncertainty
- **Target**: 25,000-50,000 additional configurations

### Phase 3: Physics-Informed Sampling
- **QI-focused**: Emphasize quasi-isodynamic configurations
- **Optimization Trajectories**: Include optimization paths
- **Failure Cases**: Document convergence failures
- **Target**: 15,000-25,000 specialized configurations

## Expected Performance

### Dataset Scale
- **Total Configurations**: 200,000-300,000
- **Successful Simulations**: >95% convergence rate
- **Parameter Coverage**: Comprehensive coverage of feasible space

### ML Model Performance
- **Accuracy**: <1% RMSE on key physics metrics
- **Speed**: 1000x+ faster than VMEC++ simulation
- **Reliability**: Robust predictions across parameter space

### Computational Requirements
- **Generation Time**: 2-4 weeks with 8-16 parallel workers
- **Storage**: 100-200 GB for complete dataset
- **Training Time**: 1-4 hours for ML model training

## Usage Examples

### Generate Custom Parameter Set

```python
from synthetic_data_generator import SyntheticDataGenerator, SyntheticDataConfig

# Custom configuration
config = SyntheticDataConfig(
    aspect_ratio_range=(3.0, 6.0),
    elongation_range=(0.5, 1.5),
    n_systematic_samples=5000,
    n_lhs_samples=10000,
    output_dir="./custom_data"
)

# Generate data
generator = SyntheticDataGenerator(config)
generator.generate_all_data()
```

### Train and Use Surrogate Model

```python
from ml_surrogate_model import StellaratorSurrogateModel
import numpy as np

# Load and train model
surrogate = StellaratorSurrogateModel("./demo_synthetic_data", "random_forest")
df = surrogate.load_data()
X, y = surrogate.prepare_features_and_targets(df)
surrogate.train_model(X, y)

# Make predictions
input_params = np.array([[4.0, 1.2, 0.5, 3, 0.1]])  # [aspect_ratio, elongation, ...]
predictions = surrogate.predict(input_params)

print("Predicted physics metrics:", predictions)
```

### Compare Multiple Models

```python
from ml_surrogate_model import compare_models

# Compare different ML algorithms
results = compare_models(
    data_path="./demo_synthetic_data",
    model_types=["random_forest", "xgboost", "neural_network"]
)
```

## Key Benefits

### 1. **Open Source Foundation**
- All simulators are open source and freely available
- No licensing restrictions for commercial use
- Community-driven development and support

### 2. **Scalable Data Generation**
- Automated pipeline for large-scale data generation
- Parallel processing for efficiency
- Robust error handling and quality control

### 3. **High-Performance ML Models**
- Multiple algorithms optimized for physics simulation
- 1000x+ speed improvement over traditional simulation
- Uncertainty quantification for reliable predictions

### 4. **Physics-Informed Design**
- Parameter ranges based on realistic stellarator designs
- Focus on quasi-isodynamic configurations
- Integration with existing optimization frameworks

## Advanced Features

### Multi-Fidelity Training
```python
# Generate data at different fidelity levels
config_low = ConstellarationSettings(
    vmec_preset_settings=VmecPresetSettings(fidelity="low_fidelity")
)
config_high = ConstellarationSettings(
    vmec_preset_settings=VmecPresetSettings(fidelity="high_fidelity")
)
```

### Uncertainty Quantification
```python
# Train ensemble models for uncertainty estimation
from sklearn.ensemble import RandomForestRegressor

# Multiple models with different random seeds
models = [RandomForestRegressor(random_state=i) for i in range(10)]
predictions = [model.predict(X_test) for model in models]
uncertainty = np.std(predictions, axis=0)
```

### Real-Time Optimization
```python
# Use surrogate model in optimization loop
from scipy.optimize import minimize

def objective(params):
    prediction = surrogate.predict(params.reshape(1, -1))
    return prediction[0, target_index]  # Minimize specific metric

result = minimize(objective, initial_guess, method='L-BFGS-B')
```

## Contributing

We welcome contributions to improve the synthetic data generation pipeline:

1. **Parameter Space Extensions**: Add new physics parameters
2. **Sampling Algorithms**: Implement advanced sampling techniques
3. **ML Models**: Add new machine learning architectures
4. **Validation**: Improve data quality and validation methods

## Citation

If you use this synthetic data generation framework in your research, please cite:

```bibtex
@article{cadena2025constellaration,
  title={ConStellaration: A dataset of QI-like stellarator plasma boundaries and optimization benchmarks},
  author={Cadena, Santiago A and Merlo, Andrea and Laude, Emanuel and others},
  journal={arXiv preprint arXiv:2506.19583},
  year={2025}
}
```

## License

This project is released under the MIT License, consistent with the underlying VMEC++ and ConStellaration frameworks.

## Support

For questions and support:
- Open an issue in this repository
- Check the ConStellaration documentation
- Join the fusion ML community discussions

---

**Note**: This framework demonstrates how open-source physics simulators can be leveraged to create large-scale synthetic datasets for machine learning. The approach is scalable and can be adapted to other physics simulation domains beyond stellarator design.