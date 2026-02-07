from abc import ABC, abstractmethod
from typing import Any, List


class IPostProcessor(ABC):
    """
    Interface for all routing post-processors.
    """

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
