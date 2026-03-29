"""
Evolution strategies for incorporating local search improvements in Differential Evolution.

This module implements the Strategy Pattern to support different approaches for
handling local search optimization within the DE framework.

Strategies:
    - Baldwinian Evolution: Phenotypic learning - improved fitness is used for selection,
      but genetic material (continuous vector) remains unchanged.
    - Lamarckian Evolution: Genotypic learning - genetic material is updated to reflect
      the improvements discovered by local search.

Theoretical Background:
    Baldwin (1896) proposed that learning can guide evolution without directly modifying
    genes. In computational optimization, Baldwinian evolution means:
        - Local search improves solution quality (phenotype)
        - Selection is based on improved fitness
        - Genetic encoding (genotype) is unchanged

    Lamarck (1809) proposed that acquired characteristics can be inherited. In
    computational optimization, Lamarckian evolution means:
        - Local search improves solution quality
        - Genetic encoding is reverse-engineered from improved solution
        - Offspring inherit the learned improvements

References:
    Whitley, D., Gordon, V.S., & Mathias, K. (1994). "Lamarckian Evolution, the
    Baldwin Effect and Function Optimization." In PPSN III: Proceedings of the
    International Conference on Evolutionary Computation, 5-15.

    Houck, C.R., Joines, J.A., & Kay, M.G. (1996). "Comparison of genetic algorithms,
    random restart and two-opt switching for solving large location-allocation problems."
    Computers & Operations Research, 23(6), 587-596.
"""

from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np


class EvolutionStrategy(ABC):
    """
    Abstract base class for evolution strategies.

    Defines the interface for determining which continuous vector survives
    the selection process when local search has been applied.
    """

    @abstractmethod
    def get_surviving_vector(
        self,
        original_vector: np.ndarray,
        optimized_routes: List[List[int]],
        encoder_func: Callable[[List[List[int]]], np.ndarray],
    ) -> np.ndarray:
        """
        Determine which continuous vector should survive selection.

        Args:
            original_vector: The continuous trial vector before local search.
            optimized_routes: The discrete routes after local search optimization.
            encoder_func: Function to encode discrete routes back to continuous vector.

        Returns:
            The continuous vector that should replace the target in the population.
        """
        pass


class BaldwinianStrategy(EvolutionStrategy):
    """
    Baldwinian evolution strategy.

    The improved fitness from local search is used for selection, but the
    genetic material (continuous Random Key vector) remains unchanged.

    Characteristics:
        - Phenotypic learning: Only observable traits improve
        - Genotypic stability: Genetic encoding is preserved
        - Slower convergence: Good solutions must be rediscovered each generation
        - Better diversity: Population maintains genetic variety

    Use When:
        - You want to maintain population diversity
        - You suspect premature convergence issues
        - Local search may produce overfitted solutions
        - Genotype-phenotype mapping is expensive
    """

    def get_surviving_vector(
        self,
        original_vector: np.ndarray,
        optimized_routes: List[List[int]],
        encoder_func: Callable[[List[List[int]]], np.ndarray],
    ) -> np.ndarray:
        """
        Return the original continuous vector unchanged.

        Args:
            original_vector: The continuous trial vector before local search.
            optimized_routes: Ignored in Baldwinian strategy.
            encoder_func: Ignored in Baldwinian strategy.

        Returns:
            The original continuous vector (genetic material unchanged).
        """
        return original_vector


class LamarckianStrategy(EvolutionStrategy):
    """
    Lamarckian evolution strategy.

    The genetic material (continuous Random Key vector) is updated to reflect
    the improvements discovered by local search.

    Characteristics:
        - Genotypic learning: Genetic code is modified by experience
        - Faster convergence: Good solutions are directly encoded
        - Potential loss of diversity: Population may converge quickly
        - Direct inheritance: Offspring start with learned improvements

    Use When:
        - You want faster convergence to high-quality solutions
        - Local search consistently produces improvements
        - Computational budget is limited
        - You have a robust encoder that preserves solution quality
    """

    def get_surviving_vector(
        self,
        original_vector: np.ndarray,
        optimized_routes: List[List[int]],
        encoder_func: Callable[[List[List[int]]], np.ndarray],
    ) -> np.ndarray:
        """
        Reverse-encode the optimized routes into a continuous vector.

        Args:
            original_vector: Ignored in Lamarckian strategy.
            optimized_routes: The discrete routes after local search optimization.
            encoder_func: Function to encode discrete routes back to continuous vector.

        Returns:
            A new continuous vector encoding the optimized solution.
        """
        return encoder_func(optimized_routes)


def create_evolution_strategy(strategy_name: str) -> EvolutionStrategy:
    """
    Factory function to create evolution strategy instances.

    Args:
        strategy_name: Name of the strategy. Options: "baldwinian", "lamarckian"

    Returns:
        An instance of the requested evolution strategy.

    Raises:
        ValueError: If strategy_name is not recognized.

    Examples:
        >>> strategy = create_evolution_strategy("lamarckian")
        >>> isinstance(strategy, LamarckianStrategy)
        True

        >>> strategy = create_evolution_strategy("baldwinian")
        >>> isinstance(strategy, BaldwinianStrategy)
        True
    """
    strategy_name_lower = strategy_name.lower().strip()

    if strategy_name_lower == "baldwinian":
        return BaldwinianStrategy()
    elif strategy_name_lower == "lamarckian":
        return LamarckianStrategy()
    else:
        raise ValueError(f"Unknown evolution strategy: '{strategy_name}'. Valid options: 'baldwinian', 'lamarckian'")
