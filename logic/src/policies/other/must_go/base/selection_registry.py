"""
Selection Registry Module.

This module provides a registry for `IMustGoSelectionStrategy` classes.
It allows dynamic registration and retrieval of strategies, facilitating
plugin-style extensions.

Attributes:
    None

Example:
    >>> from logic.src.policies.must_go.base.selection_registry import MustGoSelectionRegistry
    >>> @MustGoSelectionRegistry.register("my_strategy")
    >>> class MyStrategy(IMustGoSelectionStrategy): ...
"""

from typing import Dict, Optional, Type

from logic.src.interfaces.must_go import IMustGoSelectionStrategy


class MustGoSelectionRegistry:
    """Registry for Must Go selection strategies."""

    _strategies: Dict[str, Type[IMustGoSelectionStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy."""

        def wrapper(strategy_cls: Type[IMustGoSelectionStrategy]):
            """Register the class with the given name."""
            cls._strategies[name.lower()] = strategy_cls
            return strategy_cls

        return wrapper

    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type[IMustGoSelectionStrategy]]:
        """Get a strategy class by name."""
        return cls._strategies.get(name.lower())
