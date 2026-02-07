"""
Base abstract class and simple baseline implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from logic.src.utils.logging.pylogger import get_pylogger
from tensordict import TensorDict

logger = get_pylogger(__name__)


class Baseline(nn.Module, ABC):
    """Base class for baselines."""

    def __init__(self):
        """Initialize the baseline."""
        super().__init__()

    @abstractmethod
    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """Compute baseline value."""
        raise NotImplementedError

    def unwrap_batch(self, batch: Any) -> Tuple[Any, Optional[torch.Tensor]]:
        """Unwrap the batch if it's wrapped with baseline values.

        Args:
            batch: Wrapped batch.

        Returns:
            Tuple: Unwrapped batch data and optional baseline.
        """
        if isinstance(batch, (dict, TensorDict)):
            if "data" in list(batch.keys()) and "baseline" in list(batch.keys()):
                return batch["data"], batch["baseline"]
        return batch, None

    def unwrap_dataset(self, dataset: Any) -> Any:
        """Unwrap the dataset if it's wrapped."""
        from logic.src.data.datasets import BaselineDataset

        if isinstance(dataset, BaselineDataset):
            return dataset.dataset
        return dataset

    def epoch_callback(
        self,
        policy: nn.Module,
        epoch: int,
        val_dataset: Optional[Any] = None,
        env: Optional[Any] = None,
    ):
        """Optional callback at epoch end."""
        pass

    def wrap_dataset(
        self,
        dataset: Any,
        policy: Optional[nn.Module] = None,
        env: Optional[Any] = None,
    ) -> Any:
        """Optional wrap dataset."""
        return dataset

    def setup(self, policy: nn.Module):
        """Optional setup with policy reference."""
        pass

    def get_learnable_parameters(self) -> list:
        """Get learnable parameters for the optimizer."""
        return []


class NoBaseline(Baseline):
    """No baseline (vanilla REINFORCE)."""

    def __init__(self, **kwargs):
        """Initialize NoBaseline."""
        super().__init__()

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Return zero baseline (no variance reduction).

        Args:
            td: TensorDict with environment state (unused).
            reward: Current batch rewards.
            env: Environment (unused).

        Returns:
            torch.Tensor: Zeros matching the reward shape.
        """
        return torch.zeros_like(reward)


class MeanBaseline(Baseline):
    """
    Simple batch-mean baseline.

    Uses the mean reward of the current batch as the baseline value.
    Zero computational overhead but limited variance reduction.
    Useful as a simple default or for debugging.
    """

    def __init__(self, **kwargs):
        """Initialize MeanBaseline."""
        super().__init__()

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline as mean of current batch rewards.

        Args:
            td: TensorDict with environment state (unused).
            reward: Current batch rewards.
            env: Environment (unused).

        Returns:
            torch.Tensor: Mean reward expanded to match reward shape.
        """
        return (
            reward.mean(dim=0, keepdim=True).expand_as(reward) if reward.dim() > 1 else reward.mean().expand_as(reward)
        )
