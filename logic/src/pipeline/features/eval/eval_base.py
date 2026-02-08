"""
Base class for evaluation strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


class EvalBase(ABC):
    """Base class for evaluation strategies."""

    def __init__(self, env: Any, progress: bool = True, device: str | torch.device = "cpu", **kwargs):
        """Initialize Class.

        Args:
            env (Any): Description of env.
            progress (bool): Description of progress.
            device (str | torch.device): Description of device.
            kwargs (Any): Description of kwargs.
        """
        self.env = env
        self.progress = progress
        self.device = device

    @abstractmethod
    def __call__(self, policy: Any, data_loader: DataLoader, return_results: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Evaluate policy on dataset.

        Args:
            policy: Policy to evaluate
            data_loader: DataLoader with test data

        Returns:
            Dict with metrics (reward, etc.)
        """
        pass
