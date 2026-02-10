"""Config-like objects supporting nested traversal.

This module defines the ITraversable protocol for duck typing with
configuration objects (dict, DictConfig, dataclass).

Example:
    >>> from logic.src.interfaces.traversable import ITraversable
    >>> def extract_param(config: ITraversable, key: str) -> Any:
    ...     if key in config:
    ...         return config[key]
    ...     return config.get(key, None)
"""

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class ITraversable(Protocol):
    """Config-like objects supporting nested traversal.

    This protocol unifies dict, DictConfig (OmegaConf), and dataclass objects
    that need to be traversed in configuration parsing. It eliminates repeated
    isinstance checks for dict/DictConfig/list patterns.

    **Replaces patterns like**:
        - isinstance(config, dict) or isinstance(config, DictConfig)
        - hasattr(config, '__getitem__') and hasattr(config, 'get')

    Example:
        >>> def extract_param(config: ITraversable, key: str) -> Any:
        ...     if key in config:
        ...         return config[key]
        ...     return config.get(key, None)
    """

    def __getitem__(self, key: str) -> Any:
        """Get value by key.

        Args:
            key: Configuration key

        Returns:
            Configuration value

        Raises:
            KeyError: If key not found
        """
        ...

    def __contains__(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Configuration key

        Returns:
            True if key exists
        """
        ...

    def keys(self) -> Any:
        """Return available configuration keys.

        Returns:
            Iterable of key names
        """
        ...

    def items(self) -> Any:
        """Return key-value pairs.

        Returns:
            Iterable of (key, value) tuples
        """
        ...

    def values(self) -> Any:
        """Return all values.

        Returns:
            Iterable of values
        """
        ...

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get value by key with optional default.

        Args:
            key: Configuration key
            default: Value to return if key not found

        Returns:
            Configuration value or default
        """
        ...
