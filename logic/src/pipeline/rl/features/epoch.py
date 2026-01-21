"""
Epoch-level utilities for RL training.
"""
from typing import Optional

import torch.nn as nn
from torch.utils.data import Dataset

from logic.src.data.datasets import GeneratorDataset


def prepare_epoch(
    model: nn.Module, env: any, baseline: any, dataset: Dataset, epoch: int, phase: str = "train"
) -> Dataset:
    """
    Prepare dataset for a new epoch.
    Handles baseline wrapping for RolloutBaseline.
    """
    if phase == "train" and hasattr(baseline, "wrap_dataset"):
        # Wrap dataset with baseline values (e.g. RolloutBaseline)
        return baseline.wrap_dataset(model, dataset, env)
    return dataset


def regenerate_dataset(
    env: any,
    size: int,
) -> Optional[Dataset]:
    """
    Regenerate training dataset using environment generator.
    """
    if hasattr(env, "generator"):
        return GeneratorDataset(env.generator, size)
    return None
