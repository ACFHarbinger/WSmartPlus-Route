"""
Selection Registry Module.

This module provides a registry for `IMandatorySelectionStrategy` classes.
It allows dynamic registration and retrieval of strategies, facilitating
plugin-style extensions.

Attributes:
    MandatorySelectionRegistry: Registry for strategy classes.

Example:
    >>> from logic.src.policies.mandatory.base.selection_registry import MandatorySelectionRegistry
    >>> @MandatorySelectionRegistry.register("my_strategy")
    >>> class MyStrategy(IMandatorySelectionStrategy): ...
"""

from typing import Dict, Optional, Type

from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy


class MandatorySelectionRegistry:
    """Registry for Mandatory selection strategies.

    Attributes:
        _strategies (Dict[str, Type[IMandatorySelectionStrategy]]): Internal mapping of names to strategy classes.

    Example:
        >>> @MandatorySelectionRegistry.register("my_strategy")
        >>> class MyStrategy(IMandatorySelectionStrategy):
        ...     pass
    """

    _strategies: Dict[str, Type[IMandatorySelectionStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy.

        Args:
            name (str): Unique name to register the strategy under.

        Returns:
            Callable: A decorator that registers the class.
        """

        def wrapper(strategy_cls: Type[IMandatorySelectionStrategy]):
            """Register the class with the given name."""
            cls._strategies[name.lower()] = strategy_cls
            return strategy_cls

        return wrapper

    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type[IMandatorySelectionStrategy]]:
        """Get a strategy class by name.

        Args:
            name (str): Name of the strategy to retrieve.

        Returns:
            Optional[Type[IMandatorySelectionStrategy]]: The strategy class if found, else None.
        """
        return cls._strategies.get(name.lower())

    @classmethod
    def list_strategies(cls) -> list:
        """List all registered strategy names.

        Returns:
            List[str]: List of registered strategy names.
        """
        return list(cls._strategies.keys())
