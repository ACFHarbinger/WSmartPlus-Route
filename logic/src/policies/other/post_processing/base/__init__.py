"""
Routing Post-Processing Base Package.

This package defines the core infrastructure for the "Routing Post-Processing"
policy, including the factory pattern for algorithm creation,
and the registry for available algorithms.

Attributes:
    PostProcessorFactory (class): Factory for creating algorithms.
    PostProcessorRegistry (class): Registry for algorithm classes.

Example:
    >>> from logic.src.policies.other.post_processing.base import PostProcessorFactory
    >>> factory = PostProcessorFactory()
    >>> algorithm = factory.create("fast_tsp")
"""

from logic.src.interfaces import IPostProcessor

from .factory import PostProcessorFactory
from .registry import PostProcessorRegistry

__all__ = [
    "IPostProcessor",
    "PostProcessorRegistry",
    "PostProcessorFactory",
]
