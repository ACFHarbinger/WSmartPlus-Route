"""
Post-Processing Registry Module.

This module provides a registry for post-processing operators. It allows
registering and retrieving post-processors by name.

Attributes:
    PostProcessorRegistry (class): The registry class.

Example:
    >>> from logic.src.policies.post_processing.registry import PostProcessorRegistry
    >>> PostProcessorRegistry.register("my_processor", MyProcessorClass)
    >>> cls = PostProcessorRegistry.get("my_processor")
"""

from typing import Callable, Dict, Optional, Type

from logic.src.interfaces.post_processing import IPostProcessor


class PostProcessorRegistry:
    """Registry for routing post-processing strategies."""

    _strategies: Dict[str, Type[IPostProcessor]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a post-processor."""

        def wrapper(processor_cls: Type[IPostProcessor]):
            """Wrapper for registering the processor class."""
            cls._strategies[name.lower()] = processor_cls
            return processor_cls

        return wrapper

    @classmethod
    def get(cls, name: str) -> Optional[Type[IPostProcessor]]:
        """Retrieve a post-processor by name."""
        return cls._strategies.get(name.lower())
