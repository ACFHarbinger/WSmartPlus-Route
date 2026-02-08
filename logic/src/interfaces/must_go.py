from typing import List, Protocol, runtime_checkable

from logic.src.policies.must_go.base.selection_context import SelectionContext


@runtime_checkable
class MustGoSelectionStrategy(Protocol):
    """Interface for Must Go selection strategies."""

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Determine which bins must be collected based on the strategy logic.
        """
        ...
