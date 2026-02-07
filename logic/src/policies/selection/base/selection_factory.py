from typing import Optional, Type, cast

from .selection_registry import MustGoSelectionRegistry
from .selection_strategy import MustGoSelectionStrategy


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
        from ..selection_combined import CombinedSelection
        from ..selection_last_minute import LastMinuteSelection
        from ..selection_lookahead import LookaheadSelection
        from ..selection_regular import RegularSelection
        from ..selection_revenue import RevenueThresholdSelection
        from ..selection_service_level import ServiceLevelSelection

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
            try:
                return cls(**kwargs)
            except TypeError:
                # Fallback for strategies that don't accept args
                return cls()

        # Check default map
        cls_def = cast(Optional[Type[MustGoSelectionStrategy]], default_map.get(name.lower()))
        if cls_def:
            try:
                return cls_def(**kwargs)
            except TypeError:
                # Fallback for strategies that don't accept args
                return cls_def()

        raise ValueError(f"Unknown selection strategy: {name}")
