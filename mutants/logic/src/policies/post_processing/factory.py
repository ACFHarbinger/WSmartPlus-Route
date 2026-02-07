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
