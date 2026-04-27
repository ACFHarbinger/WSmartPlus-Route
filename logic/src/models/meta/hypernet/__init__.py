"""HyperNetwork for Adaptive Reward Weighting.

This package implements a metadata-driven approach to adjust the relative
importance of multiple objectives (e.g., overflows vs distance) in the
reward function, adapting to temporal changes and fleet performance.

Attributes:
    HyperNetwork: Neural generator for adaptive weighting scalars.
    HyperNetworkOptimizer: Manager for hypernetwork training and inference.

Example:
    None
"""

from .hypernetwork import HyperNetwork as HyperNetwork
from .optimizer import HyperNetworkOptimizer as HyperNetworkOptimizer

__all__ = [
    "HyperNetwork",
    "HyperNetworkOptimizer",
]
