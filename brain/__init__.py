"""
Entity Brain - Custom Transformer Model

This module implements a from-scratch Transformer language model
for the Entity project. No pre-trained models are used.
"""

from .model import EntityTransformer, EntityConfig
from .tokenizer import EntityTokenizer, BPETokenizer
from .inference import InferenceEngine, SamplingConfig

__all__ = [
    "EntityTransformer",
    "EntityConfig",
    "EntityTokenizer",
    "BPETokenizer",
    "InferenceEngine",
    "SamplingConfig",
]

__version__ = "0.1.0"