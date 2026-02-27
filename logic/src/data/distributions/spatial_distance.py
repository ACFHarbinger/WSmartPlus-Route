"""
Spatial sampling distributions - Distance-based.
"""

from typing import Any, Tuple

import numpy as np


class Distance:
    """Distance-based sampling."""

    def __init__(self, graph: Tuple[Any, Any]):
        """Initialize Class.

        Args:
            graph (Tuple[Any, Any]): (depot, loc) coordinates.
        """
        self.depot, self.loc = graph

    def sample_array(self) -> np.ndarray:
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
