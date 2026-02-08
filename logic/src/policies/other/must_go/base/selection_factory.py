"""
Selection Factory Module.

This module implements the Factory pattern for creating `MustGoSelectionStrategy`
instances. It allows creating strategies by name or from a configuration object.

Attributes:
    None

Example:
    >>> from logic.src.policies.must_go.base.selection_factory import MustGoSelectionFactory
    >>> strategy = MustGoSelectionFactory.create_strategy("regular", threshold=2)
"""

from typing import Any, Optional, Type, cast

from logic.src.interfaces.must_go import IMustGoSelectionStrategy

from .selection_registry import MustGoSelectionRegistry


class MustGoSelectionFactory:
    """Factory for creating Must Go selection strategies."""

    @staticmethod
    def create_strategy(name: str, **kwargs) -> IMustGoSelectionStrategy:
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
        cls_def = cast(Optional[Type[IMustGoSelectionStrategy]], default_map.get(name.lower()))
        if cls_def:
            try:
                return cls_def(**kwargs)
            except TypeError:
                # Fallback for strategies that don't accept args
                return cls_def()

        raise ValueError(f"Unknown selection strategy: {name}")

    @classmethod
    def create_from_config(cls, config: Any) -> IMustGoSelectionStrategy:
        """
        Create a selection strategy from a MustGoConfig object.

        Args:
            config: MustGoConfig instance.
        """
        if config.strategy is None:
            # Default to all if no strategy specified? Or raise?
            # MustGoSelectionAction currently handles None by skipping.
            raise ValueError("No strategy specified in MustGoConfig")

        # Map config fields to strategy kwargs
        # This mapping depends on how strategies use their parameters.
        # Based on current implementation, many use 'threshold' from Context,
        # but some might take params in __init__.
        params = config.params.copy()

        if config.strategy == "last_minute":
            params["threshold"] = config.threshold
        elif config.strategy == "regular":
            params["threshold"] = config.frequency  # RegularSelection uses threshold as frequency
        elif config.strategy == "service_level":
            params["threshold"] = config.confidence_factor
        elif config.strategy == "revenue":
            params.update(
                {
                    "revenue_kg": config.revenue_kg,
                    "bin_capacity": config.bin_capacity,
                    "revenue_threshold": config.revenue_threshold,
                }
            )
        elif config.strategy == "combined":
            params.update({"strategies": config.combined_strategies, "logic": config.logic})

        return cls.create_strategy(config.strategy, **params)
