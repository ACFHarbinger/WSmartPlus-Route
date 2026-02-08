"""
Must-Go Base Package.

This package defines the core infrastructure for the "Must-Go" selection
policy, including the context object, factory pattern for strategy creation,
and the registry for available strategies.

Attributes:
    MustGoSelectionFactory (class): Factory for creating strategies.
    MustGoSelectionRegistry (class): Registry for strategy classes.
    SelectionContext (class): Data container for selection context.

Example:
    >>> from logic.src.policies.must_go.base import SelectionContext
    >>> ctx = SelectionContext(bin_ids=..., current_fill=...)
"""

from .selection_context import SelectionContext
from .selection_factory import MustGoSelectionFactory
from .selection_registry import MustGoSelectionRegistry

__all__ = ["MustGoSelectionFactory", "MustGoSelectionRegistry", "SelectionContext"]
