"""
PyTorch dataset classes for WSmart-Route.

Attributes:
    BaselineDataset: BaseLine dataset for training.
    ExtraKeyDataset: Extra Key dataset for training.
    TensorDictDatasetFastGeneration: TensorDict dataset for fast generation.
    FastTdDataset: Fast TensorDict dataset for training.
    GeneratorDataset: Generator dataset for training.
    TensorDictDataset: TensorDict dataset for training.

Example:
    >>> from logic.src.data.datasets import BaselineDataset
    >>> dataset = BaselineDataset(**td_kwargs)
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=tensordict_collate_fn)
    >>> for batch in dataloader:
    ...     print(batch)
    ...     break
"""

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
]
