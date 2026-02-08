"""
Selection Registry Module.

This module provides a registry for `MustGoSelectionStrategy` classes.
It allows dynamic registration and retrieval of strategies, facilitating
plugin-style extensions.

Attributes:
    None

Example:
    >>> from logic.src.policies.must_go.base.selection_registry import MustGoSelectionRegistry
    >>> @MustGoSelectionRegistry.register("my_strategy")
    >>> class MyStrategy(MustGoSelectionStrategy): ...
"""

from typing import Dict, Optional, Type

from logic.src.interfaces.must_go import MustGoSelectionStrategy


class MustGoSelectionRegistry:
    """Registry for Must Go selection strategies."""

    _strategies: Dict[str, Type[MustGoSelectionStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy."""

        def wrapper(strategy_cls: Type[MustGoSelectionStrategy]):
            """Register the class with the given name."""
            cls._strategies[name.lower()] = strategy_cls
            return strategy_cls

        return wrapper

    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type[MustGoSelectionStrategy]]:
        """Get a strategy class by name."""
        return cls._strategies.get(name.lower())
