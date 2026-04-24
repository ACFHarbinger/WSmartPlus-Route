"""Mandatory Base Package.

This package defines the core infrastructure for the "Mandatory" selection
policy, including the context object, factory pattern for strategy creation,
and the registry for available strategies.

Attributes:
    MandatorySelectionFactory: Factory for creating strategy instances.
    MandatorySelectionRegistry: Registry for mapping names to strategy classes.
    SelectionContext: Data container for selection context and state.

Example:
    >>> from logic.src.policies.mandatory_selection.base import SelectionContext
    >>> ctx = SelectionContext(bin_ids=np.array([0, 1]), current_fill=np.array([50, 95]))
"""

from .selection_context import SelectionContext
from .selection_factory import MandatorySelectionFactory
from .selection_registry import MandatorySelectionRegistry

__all__ = ["MandatorySelectionFactory", "MandatorySelectionRegistry", "SelectionContext"]
