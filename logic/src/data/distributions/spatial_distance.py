"""
Spatial sampling distributions - Distance-based.
"""

from typing import Any, Optional, Tuple

import numpy as np
import torch

from .base import BaseDistribution


class Distance(BaseDistribution):
    """Distance-based sampling."""

    def __init__(self, graph: Tuple[Any, Any]):
        """Initialize Class.

        Args:
            graph (Tuple[Any, Any]): (depot, loc) coordinates.
        """
        self.depot, self.loc = graph

    def _sample_array(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample from distance-based distribution.

        Returns:
            np.ndarray: Sampled values [num_loc] or [batch_size, num_loc]
        """
        if self.depot.ndim == 1:
            # Single instance: depot is [coords], loc is [num_loc, coords]
            # Reshape for broadcasting
            wp = np.linalg.norm(self.depot[None, :] - self.loc, axis=-1)
            return (1 + (wp / wp.max() * 99).astype(int)) / 100.0
        else:
            # Batch: depot is [batch_size, coords], loc is [batch_size, num_loc, coords]
            # Matching generation.py's depot[:, None, :] - loc
            wp = np.linalg.norm(self.depot[:, None, :] - self.loc, axis=-1)
            return (1 + (wp / wp.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.0

    def _sample_tensor(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """PyTorch version of distance-based sampling."""
        # Ensure inputs are tensors (standardizing in case they are still numpy)
        depot = torch.as_tensor(self.depot).float()
        loc = torch.as_tensor(self.loc).float()
        if depot.ndim == 1:
            # Single instance: depot [2], loc [num_loc, 2]
            # Using [None, :] is equivalent to .unsqueeze(0)
            diff = depot[None, :] - loc
            wp = torch.linalg.vector_norm(diff, dim=-1)

            # Logic: (1 + int(dist/max * 99)) / 100
            # .to(torch.int) handles the floor truncation for positive numbers
            wp_max = wp.max()
            scaled = 1 + (wp / wp_max * 99).to(torch.int)
            return scaled.float() / 100.0
        else:
            # Batch: depot [batch_size, 2], loc [batch_size, num_loc, 2]
            # Broadcasts to [batch_size, num_loc, 2]
            diff = depot[:, None, :] - loc
            wp = torch.linalg.vector_norm(diff, dim=-1)

            # Max along the location dimension (axis=-1)
            wp_max = wp.max(dim=-1, keepdim=True).values
            scaled = 1 + (wp / wp_max * 99).to(torch.int)
            return scaled.float() / 100.0
