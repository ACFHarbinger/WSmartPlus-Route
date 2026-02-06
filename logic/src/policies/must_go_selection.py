"""
Must Go Selection subsystem.

This module provides a decoupled way to identify which bins MUST be collected
before running routing optimization. It uses the Strategy Pattern to allow
different selection logics (periodic, threshold-based, predictive, etc.) to be
swapped dynamically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import numpy as np
from numpy.typing import NDArray


@dataclass
class SelectionContext:
    """
    Context container for all potential inputs required by selection strategies.
    """

    bin_ids: NDArray[np.int32]
    current_fill: NDArray[np.float64]
    accumulation_rates: Optional[NDArray[np.float64]] = None
    std_deviations: Optional[NDArray[np.float64]] = None
    current_day: int = 0
    threshold: float = 0.0
    next_collection_day: Optional[int] = None
    distance_matrix: Optional[NDArray[Any]] = None
    paths_between_states: Optional[List[List[List[int]]]] = None
    vehicle_capacity: float = 0.0
    revenue_kg: float = 0.0
    bin_density: float = 0.0
    bin_volume: float = 0.0


class MustGoSelectionStrategy(ABC):
    """Abstract Base Class for Must Go selection strategies."""

    @abstractmethod
    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Determine which bins must be collected based on the strategy logic.
        """
        pass


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


class MustGoSelectionFactory:
    """Factory for creating Must Go selection strategies."""

    @staticmethod
    def create_strategy(name: str, **kwargs) -> MustGoSelectionStrategy:
        """
        Create a selection strategy by name.

        Args:
            name: Name of the strategy.
            **kwargs: Arguments to pass to the strategy constructor.
        """
        # Lazy imports to avoid circular dependencies and keep strategies separated
        from .selection.selection_combined import CombinedSelection
        from .selection.selection_last_minute import LastMinuteSelection
        from .selection.selection_lookahead import LookaheadSelection
        from .selection.selection_regular import RegularSelection
        from .selection.selection_revenue import RevenueThresholdSelection
        from .selection.selection_service_level import ServiceLevelSelection

        default_map = {
            "service_level": ServiceLevelSelection,
            "regular": RegularSelection,
            "lookahead": LookaheadSelection,
            "last_minute": LastMinuteSelection,
            "revenue": RevenueThresholdSelection,
            "revenue_threshold": RevenueThresholdSelection,
            "combined": CombinedSelection,
        }

        # Check explicit registry first
        cls = MustGoSelectionRegistry.get_strategy_class(name)
        if cls:
            # If strategy accepts kwargs, pass them.
            # Ideally we should inspection, but let's try passing if kwargs exists.
            # Most existing strategies don't take init args, but CombinedSelection does.
            try:
                return cls(**kwargs)
            except TypeError:
                # Fallback for strategies that don't accept args
                return cls()

        # Check default map
        cls = default_map.get(name.lower())
        if cls:
            try:
                return cls(**kwargs)
            except TypeError:
                # Fallback for strategies that don't accept args
                return cls()

        raise ValueError(f"Unknown selection strategy: {name}")
