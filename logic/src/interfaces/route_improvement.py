"""route_improvement.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import route_improvement
"""

from abc import ABC, abstractmethod
from typing import Any, List


class IRouteImprovement(ABC):
    """
    Interface for all routing route improvers.
    """

    def __init__(self, **kwargs: Any):
        """Initialize route improver with config."""
        # Support both direct kwargs and passing a 'config' dict (for composability).
        if "config" in kwargs and len(kwargs) == 1:
            self.config = kwargs["config"]
        else:
            self.config = kwargs

    @abstractmethod
    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Refine a given tour.

        Args:
            tour: Initial tour (List of bin IDs including depot 0s)
            **kwargs: Context dictionary containing distance matrix, etc.

        Returns:
            List[int]: Refined tour.
        """
        pass
