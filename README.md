# Stellarator Synthetic Data Generator

This repository contains tools for generating synthetic data from stellarator fusion simulators and training machine learning models to mimic their behavior. The goal is to create surrogate models that can rapidly approximate the results of computationally expensive physics simulations.

## Overview

Stellarators are a type of fusion device designed to confine hot plasma using twisted, three-dimensional magnetic fields to sustain nuclear fusion reactions. Simulating stellarator behavior is computationally expensive, making design optimization challenging. This project aims to:

1. Generate synthetic data using the VMEC++ and ConStellaration simulators
2. Process this data for machine learning
3. Convert numerical data to text format for language model training
4. Train surrogate models (neural networks or language models) that can rapidly approximate simulation results
5. Provide tools for evaluating and using these models

## Requirements

- Python 3.10+
- VMEC++ (https://github.com/proximafusion/vmecpp)
- ConStellaration (https://github.com/proximafusion/constellaration)
- TensorFlow 2.x or PyTorch + Transformers (for language models)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Plotly

## Installation

1. Clone this repository:
```bash
git clone https://github.com/neubig/stellarator-synth.git
cd stellarator-synth
```

2. Install the required dependencies:
```bash
# System dependencies (Ubuntu)
sudo apt-get update
sudo apt-get install -y build-essential cmake libnetcdf-dev liblapack-dev libomp-dev libhdf5-dev python3-dev

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies for neural network approach
pip install numpy pandas scikit-learn matplotlib plotly tensorflow tqdm joblib

# Additional dependencies for language model approach
pip install torch transformers datasets

# Install VMEC++ and ConStellaration
pip install git+https://github.com/proximafusion/vmecpp.git
pip install git+https://github.com/proximafusion/constellaration.git
```

## Usage

### 1. Generate Synthetic Data

Generate a dataset of stellarator configurations:

```bash
python src/generate_dataset.py --n_samples 100 --output_dir data/raw_dataset --max_workers 4 --optimization_time 60
```

Parameters:
- `--n_samples`: Number of stellarator configurations to generate
- `--output_dir`: Directory to save the generated data
- `--max_workers`: Number of parallel workers for data generation
- `--optimization_time`: Time limit (seconds) for each optimization
- `--max_poloidal_mode`: Maximum poloidal mode number for boundary representation
- `--max_toroidal_mode`: Maximum toroidal mode number for boundary representation

### 2. Process the Dataset

Process the raw data for machine learning:

```bash
python src/process_dataset.py --input_csv data/raw_dataset/inputs.csv --output_csv data/raw_dataset/outputs.csv --output_dir data/ml_dataset
```

Parameters:
- `--input_csv`: Path to the input features CSV
- `--output_csv`: Path to the output features CSV
- `--output_dir`: Directory to save the processed data
- `--test_size`: Fraction of data to use for testing
- `--val_size`: Fraction of data to use for validation

### 3. Convert Data to Text Format (for Language Models)

Convert the numerical data to text format for language model training:

```bash
python src/convert_to_text.py --input_csv data/raw_dataset/inputs.csv --output_csv data/raw_dataset/outputs.csv --boundary_dir data/raw_dataset --equilibrium_dir data/raw_dataset --output_dir data/text_dataset --format prompt
```

Parameters:
- `--input_csv`: Path to input features CSV
- `--output_csv`: Path to output features CSV
- `--boundary_dir`: Directory containing boundary JSON files
- `--equilibrium_dir`: Directory containing equilibrium JSON files
- `--output_dir`: Directory to save text files
- `--format`: Format type ("separate", "combined", or "prompt")
- `--max_samples`: Maximum number of samples to process

### 4. Train Models

#### Neural Network Approach

Train a neural network to mimic the simulator:

```bash
python src/train_model.py --data_dir data/ml_dataset --output_dir models/surrogate_model --batch_size 32 --epochs 200 --patience 20 --layer_sizes "256,128,64"
```

Parameters:
- `--data_dir`: Directory with processed data
- `--output_dir`: Directory to save model and results
- `--batch_size`: Batch size for training
- `--epochs`: Maximum number of epochs
- `--patience`: Patience for early stopping
- `--layer_sizes`: Hidden layer sizes (comma-separated)

#### Language Model Approach

Train a language model on the text data:

```bash
# For continuous text training
python src/train_language_model.py --mode text --text_file data/text_dataset/all_samples.txt --model_name gpt2 --output_dir models/language_model --batch_size 4 --num_epochs 3

# For prompt-completion training
python src/train_language_model.py --mode prompt --prompt_file data/text_dataset/prompts.txt --completion_file data/text_dataset/completions.txt --model_name gpt2 --output_dir models/language_model --batch_size 4 --num_epochs 3
```

Parameters:
- `--mode`: Training mode ("text" or "prompt")
- `--text_file`: Path to text file (for "text" mode)
- `--prompt_file`: Path to prompts file (for "prompt" mode)
- `--completion_file`: Path to completions file (for "prompt" mode)
- `--model_name`: Pre-trained model name
- `--output_dir`: Directory to save the model
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate
- `--num_epochs`: Number of training epochs
- `--max_length`: Maximum sequence length

### 5. Run Complete Pipeline

Run the complete pipeline with a single command:

```bash
python run_pipeline.py --n_samples 10 --raw_data_dir data/raw_dataset --ml_data_dir data/ml_dataset --model_dir models/surrogate_model
```

## Data Format

The generated dataset includes:

- **Input features**: Boundary surface Fourier coefficients and configuration parameters
- **Output features**: Physics metrics like aspect ratio, elongation, rotational transform, etc.
- **Raw data**: Original boundary and equilibrium data for each configuration
- **Visualizations**: Boundary surface plots for selected configurations
- **Text data**: Human-readable text descriptions of stellarator configurations and their properties

## Model Architectures

### Neural Network Surrogate

The default neural network surrogate model is a feed-forward network with:
- Multiple hidden layers with ReLU activation
- Batch normalization and dropout for regularization
- Linear output layer for regression
- Adam optimizer with mean squared error loss

### Language Model Surrogate

The language model approach uses a pre-trained transformer model (e.g., GPT-2) fine-tuned on stellarator data:
- Can be trained on continuous text descriptions of stellarator configurations
- Can be trained on prompt-completion pairs for conditional generation
- Provides human-readable outputs that can be parsed for numerical values
- Enables natural language interaction with the surrogate model

## License

MIT License

## Acknowledgments

This project uses the following open-source software:
- [VMEC++](https://github.com/proximafusion/vmecpp) - A Python-friendly reimplementation of the Variational Moments Equilibrium Code
- [ConStellaration](https://github.com/proximafusion/constellaration) - A dataset and tools for stellarator optimization
- [Transformers](https://github.com/huggingface/transformers) - State-of-the-art Natural Language Processing library

## Citation

If you use this code in your research, please cite:
```
@misc{stellarator-synth,
  author = {Neubig, Graham},
  title = {Stellarator Synthetic Data Generator},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/neubig/stellarator-synth}
}
```