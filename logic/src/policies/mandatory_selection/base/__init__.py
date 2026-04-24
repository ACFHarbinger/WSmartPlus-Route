"""Mandatory Base Package.

This package defines the core infrastructure for the "Mandatory" selection
policy, including the factory pattern for strategy creation,
and the registry for available strategies.

Attributes:
    MandatorySelectionFactory: Factory for creating strategy instances.
    MandatorySelectionRegistry: Registry for mapping names to strategy classes.

Example:
    >>> from logic.src.policies.mandatory_selection.base import MandatorySelectionFactory
    >>> factory = MandatorySelectionFactory()
    >>> strategy = factory.get_strategy("some_strategy")
"""

from .selection_factory import MandatorySelectionFactory
from .selection_registry import MandatorySelectionRegistry

__all__ = ["MandatorySelectionFactory", "MandatorySelectionRegistry"]
