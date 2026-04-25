"""Base classes for distance computation strategies.

Attributes:
    DistanceStrategy: Base class for distance computation strategies.
    IterativeDistanceStrategy: Base class for iterative distance computation strategies.

Example:
    >>> from logic.src.data.network.base import DistanceStrategy, IterativeDistanceStrategy
    >>> print(DistanceStrategy)
    <class 'logic.src.data.network.base.distance_strategy.DistanceStrategy'>
    >>> print(IterativeDistanceStrategy)
    <class 'logic.src.data.network.base.iterative_distance_strategy.IterativeDistanceStrategy'>
"""

from .distance_strategy import DistanceStrategy
from .iterative_distance_strategy import IterativeDistanceStrategy

__all__ = [
    "DistanceStrategy",
    "IterativeDistanceStrategy",
]
