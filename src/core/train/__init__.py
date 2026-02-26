"""
Training module for model creation and training.
"""

from .model import get_model, get_lightweight_model, get_small_dataset_model, get_callbacks
from .training_model import training_model
from .evaluate_model import evaluate_model