from abc import ABC, abstractmethod
from typing import List

from .selection_context import SelectionContext


class MustGoSelectionStrategy(ABC):
    """Abstract Base Class for Must Go selection strategies."""

    @abstractmethod
    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Determine which bins must be collected based on the strategy logic.
        """
        pass
