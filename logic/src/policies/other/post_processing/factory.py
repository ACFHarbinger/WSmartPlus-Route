"""
Post-Processing Factory Module.

This module implements the Factory pattern for creating post-processing operators.
It handles the instantiation of post-processors based on configuration or names.

Attributes:
    PostProcessorFactory (class): The factory class.

Example:
    >>> from logic.src.policies.post_processing.factory import PostProcessorFactory
    >>> processors = PostProcessorFactory.create_from_config(config)
"""

from typing import Any, List

from logic.src.interfaces.post_processing import IPostProcessor

from .registry import PostProcessorRegistry


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

    @classmethod
    def create_from_config(cls, config: Any) -> List[IPostProcessor]:
        """
        Create a list of post-processor instances from a PostProcessingConfig object.

        Args:
            config: PostProcessingConfig instance.
        """
        processors: List[IPostProcessor] = []
        if not config.methods:
            return processors

        for method in config.methods:
            # Map parameters based on method name and config fields
            # ClassicalLocalSearch needs 'operator_name'
            if method in ["2opt", "2opt_star", "swap", "relocate", "swap_star", "3opt"] or method == "fast_tsp":
                processor = cls.create(method)
            elif method == "ils":
                # IteratedLocalSearchPostProcessor takes many params
                # We can pass config fields if they match
                from .ils import IteratedLocalSearchPostProcessor

                processor = IteratedLocalSearchPostProcessor(
                    ls_iterations=config.iterations,
                    # Add other maps if needed
                )
            else:
                processor = cls.create(method)

            processors.append(processor)

        return processors
