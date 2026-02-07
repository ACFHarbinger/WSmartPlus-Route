"""
Combined selection strategy module.
"""

from typing import Any, Dict, List, Optional, cast

from .base.selection_context import SelectionContext
from .base.selection_factory import MustGoSelectionFactory
from .base.selection_strategy import MustGoSelectionStrategy


class CombinedSelection(MustGoSelectionStrategy):
    """
    Combines multiple selection strategies with logical OR or AND.
    """

    def __init__(
        self,
        strategies: Optional[List[Dict[str, Any]]] = None,
        combined_strategies: Optional[List[Dict[str, Any]]] = None,
        logic: str = "or",
    ):
        """
        Initialize CombinedSelection.

        Args:
            strategies: List of strategy configurations.
            combined_strategies: Alias for strategies (used in MustGoConfig).
            logic: Logical operator ('or', 'and').
        """
        self.strategy_configs = strategies or combined_strategies or []
        self.logic = logic.lower()
        if self.logic not in ["or", "and"]:
            raise ValueError(f"Unknown logic: {self.logic}. Must be 'or' or 'and'.")

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Select bins based on combined strategies.

        Args:
            context: SelectionContext.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        if not self.strategy_configs:
            return []

        # Initialize result set based on logic
        # For 'or', start empty (neutral element for union)
        # For 'and', start with all bins (neutral element for intersection) - wait,
        # actually for intersection we need the first set to intersect with.
        # So we can just process the first one and then loop.

        # Instantiate strategies
        strategies = []
        for config in self.strategy_configs:
            name = config.get("name")
            params = config.get("params", {})
            if name:
                strategy = MustGoSelectionFactory.create_strategy(name)
                strategies.append((strategy, params))

        if not strategies:
            return []

        # Execute first strategy
        first_strat, first_params = strategies[0]

        # Update context with first params
        ctx = self._update_context(context, first_params)
        result_set = set(first_strat.select_bins(ctx))
        for strategy, params in strategies[1:]:
            ctx = self._update_context(context, params)
            current_bins = set(strategy.select_bins(ctx))

            if self.logic == "or":
                result_set = result_set.union(current_bins)
            else:  # and
                result_set = result_set.intersection(current_bins)

        return list(result_set)

    def _update_context(self, context: SelectionContext, params: Dict[str, Any]) -> SelectionContext:
        """
        Create a shallow copy of context with updated parameters.
        """
        from dataclasses import replace

        # Map params to context fields
        updates = {}

        # Common mappings based on MustGoSelectionAction logic
        if "threshold" in params:
            updates["threshold"] = float(params["threshold"])

        # Add other mappings if needed
        if updates:
            return replace(context, **cast(Dict[str, Any], updates))
        return context
