"""
Base class for evaluation strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from torch.utils.data import DataLoader


class EvalBase(ABC):
    """Base class for evaluation strategies."""

    def __init__(self, env: Any, progress: bool = True, **kwargs):
        self.env = env
        self.progress = progress

    @abstractmethod
    def __call__(self, policy: Any, data_loader: DataLoader, **kwargs) -> Dict[str, float]:
        """
        Evaluate policy on dataset.

        Args:
            policy: Policy to evaluate
            data_loader: DataLoader with test data

        Returns:
            Dict with metrics (reward, etc.)
        """
        pass
