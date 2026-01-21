"""
TensorDict-based datasets for training.
"""
from __future__ import annotations

from typing import List

import torch
from tensordict import TensorDict
from torch.utils.data import Dataset


class TensorDictDataset(Dataset):
    """Dataset wrapping a TensorDict."""

    def __init__(self, data: TensorDict):
        self.data = data

    def __len__(self) -> int:
        return self.data.batch_size[0]

    def __getitem__(self, idx: int) -> TensorDict:
        return self.data[idx]


class GeneratorDataset(Dataset):
    """
    Dataset that generates instances on-the-fly.

    Useful for training with infinite data.
    """

    def __init__(self, generator, size: int):
        self.generator = generator
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> TensorDict:
        return self.generator(batch_size=1)[0]


def tensordict_collate_fn(batch: List[TensorDict]) -> TensorDict:
    """Collate list of TensorDicts into batched TensorDict."""
    return torch.stack(batch)


__all__ = [
    "TensorDictDataset",
    "GeneratorDataset",
    "tensordict_collate_fn",
]
