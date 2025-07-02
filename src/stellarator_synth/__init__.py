"""Stellarator synthetic data generation and language model training package."""

__version__ = "0.1.0"
__author__ = "OpenHands AI"
__email__ = "openhands@all-hands.dev"

from stellarator_synth.data_generator import SyntheticDataConfig, SyntheticDataGenerator
from stellarator_synth.language_model_trainer import LanguageModelTrainer
from stellarator_synth.ml_surrogate_model import MLSurrogateModel
from stellarator_synth.text_data_converter import TextDataConverter

__all__ = [
    "SyntheticDataConfig",
    "SyntheticDataGenerator", 
    "LanguageModelTrainer",
    "MLSurrogateModel",
    "TextDataConverter",
]