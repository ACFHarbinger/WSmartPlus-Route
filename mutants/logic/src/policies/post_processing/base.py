"""
Base interfaces and registry for routing post-processors.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type


class IPostProcessor(ABC):
    """
    Interface for all routing post-processors.
    """

    @abstractmethod
    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Refine a given tour.

        Args:
            tour: Initial tour (List of bin IDs including depot 0s)
            **kwargs: Context dictionary containing distance matrix, etc.

        Returns:
            List[int]: Refined tour.
        """
        pass


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


class PostProcessorFactory:
    """Factory for creating post-processing strategy instances."""

    @staticmethod
    def create(name: str) -> IPostProcessor:
        """
        Create a post-processor instance by name.
        """
        from .fast_tsp import FastTSPPostProcessor
        from .ils import IteratedLocalSearchPostProcessor
        from .local_search import ClassicalLocalSearchPostProcessor
        from .random_ls import RandomLocalSearchPostProcessor

        cls = PostProcessorRegistry.get(name)
        if not cls:
            # Fallback for dynamic/mapped names
            n_lower = name.lower()
            if n_lower == "fast_tsp":
                return FastTSPPostProcessor()
            elif n_lower in ["2opt", "2opt_star", "swap", "relocate", "swap_star", "3opt"]:
                return ClassicalLocalSearchPostProcessor(operator_name=n_lower)
            elif n_lower in ["random", "random_local_search"]:
                return RandomLocalSearchPostProcessor()
            elif n_lower == "ils":
                return IteratedLocalSearchPostProcessor()

            raise ValueError(f"Unknown post-processor: {name}")
        return cls()
