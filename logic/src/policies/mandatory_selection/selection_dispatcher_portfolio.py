"""
Portfolio Dispatcher Strategy Module.

Implements an ensemble strategy that aggregates the decisions of multiple
candidate "mandatory" strategies. It supports robust consensus by returning either
the intersection (requires all strategies to agree) or the union (requires
only one strategy to select a bin) of the candidate results.

Example:
    >>> from logic.src.policies.other.mandatory.selection_dispatcher_portfolio import PortfolioDispatcher
    >>> strategy = PortfolioDispatcher()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Set

from logic.src.interfaces.mandatory import IMandatorySelectionStrategy
from logic.src.policies.other.mandatory.base.selection_context import SelectionContext
from logic.src.policies.other.mandatory.base.selection_registry import MandatorySelectionRegistry


@MandatorySelectionRegistry.register("dispatcher_portfolio")
class PortfolioDispatcher(IMandatorySelectionStrategy):
    """
    Portfolio dispatcher for ensemble selection.
    """

    def select_bins(self, context: SelectionContext) -> List[int]:
        """
        Run multiple strategies and aggregate via union or intersection.

        Args:
            context: SelectionContext with portfolio configuration.

        Returns:
            List[int]: List of bin IDs (1-based index).
        """
        from logic.src.policies.other.mandatory.base.selection_factory import MandatorySelectionFactory

        candidates = context.dispatcher_candidate_strategies or ["last_minute", "deadline", "mip_knapsack"]
        mode = context.dispatcher_mode or "union"

        if mode not in ("union", "intersection"):
            raise ValueError(f"PortfolioDispatcher: Unknown mode '{mode}'. Use 'union' or 'intersection'.")

        results: List[Set[int]] = []
        for name in candidates:
            try:
                strategy = MandatorySelectionFactory.create_strategy(name)
                res = set(strategy.select_bins(context))
                results.append(res)
            except Exception:
                # Skip failed strategies
                continue

        if not results:
            return []

        # union or intersection
        final_set = set().union(*results) if mode == "union" else results[0].intersection(*results[1:])

        return sorted(list(final_set))
