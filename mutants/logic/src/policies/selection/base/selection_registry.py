from typing import Dict, Optional, Type

from .selection_strategy import MustGoSelectionStrategy


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
