"""mandatory.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import mandatory
"""

from typing import TYPE_CHECKING, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from logic.src.policies.mandatory_selection.base.selection_context import SelectionContext


@runtime_checkable
class IMandatorySelectionStrategy(Protocol):
    """Interface for Mandatory selection strategies."""

    def select_bins(self, context: "SelectionContext") -> List[int]:
        """
        Determine which bins must be collected based on the strategy logic.
        """
        ...
