"""
Dataset classes for WSmart-Route.
"""

# Import from utils.data.td_utils, not from a local td_utils
from logic.src.utils.data.td_utils import td_kwargs, tensordict_collate_fn

from .baseline_dataset import BaselineDataset
from .extra_key_dataset import ExtraKeyDataset
from .fast_gen_dataset import TensorDictDatasetFastGeneration
from .fast_td_dataset import FastTdDataset
from .generator_dataset import GeneratorDataset
from .td_dataset import TensorDictDataset

__all__ = [
    "BaselineDataset",
    "ExtraKeyDataset",
    "TensorDictDatasetFastGeneration",
    "FastTdDataset",
    "GeneratorDataset",
    "TensorDictDataset",
    "td_kwargs",
    "tensordict_collate_fn",
]
