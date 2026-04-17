"""
Selection Factory Module.

This module implements the Factory pattern for creating `MandatorySelectionStrategy`
instances. It allows creating strategies by name or from a configuration object.

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.base.selection_factory import MandatorySelectionFactory
    >>> strategy = MandatorySelectionFactory.create_strategy("regular", threshold=2)
"""

from typing import Any, Optional, Type, cast

from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy

from .selection_registry import MandatorySelectionRegistry


class MandatorySelectionFactory:
    """Factory for creating Mandatory selection strategies."""

    @staticmethod
    def create_strategy(name: str, **kwargs) -> IMandatorySelectionStrategy:
        """
        Create a selection strategy by name.

        Args:
            name: Name of the strategy.
            **kwargs: Arguments to pass to the strategy constructor.
        """
        # Lazy imports to avoid circular dependencies and keep strategies separated
        from ..selection_combined import CombinedSelection
        from ..selection_cvar import CVaRSelection
        from ..selection_deadline import DeadlineDrivenSelection
        from ..selection_dispatcher_portfolio import PortfolioDispatcher
        from ..selection_dispatcher_thompson import ThompsonDispatcher
        from ..selection_lagrangian import LagrangianSelection
        from ..selection_last_minute import LastMinuteSelection
        from ..selection_learned import LearnedSelection
        from ..selection_lookahead import LookaheadSelection
        from ..selection_multi_day_prob import MultiDayOverflowSelection
        from ..selection_pareto import ParetoFrontSelection
        from ..selection_profit_per_km import ProfitPerKmSelection
        from ..selection_regular import RegularSelection
        from ..selection_revenue import RevenueThresholdSelection
        from ..selection_rollout import RolloutSelection
        from ..selection_savings import SavingsSelection
        from ..selection_service_level import ServiceLevelSelection
        from ..selection_set_cover import SetCoverSelection
        from ..selection_spatial_synergy import SpatialSynergySelection
        from ..selection_stochastic_regret import StochasticRegretSelection
        from ..selection_submodular_greedy import SubmodularGreedySelection
        from ..selection_supermodular_greedy import SupermodularGreedySelection
        from ..selection_wasserstein import WassersteinRobustSelection
        from ..selection_whittle import WhittleIndexSelection

        default_map = {
            "service_level": ServiceLevelSelection,
            "regular": RegularSelection,
            "lookahead": LookaheadSelection,
            "last_minute": LastMinuteSelection,
            "revenue": RevenueThresholdSelection,
            "revenue_threshold": RevenueThresholdSelection,
            "combined": CombinedSelection,
            "deadline": DeadlineDrivenSelection,
            "multi_day_prob": MultiDayOverflowSelection,
            "pareto_front": ParetoFrontSelection,
            "profit_per_km": ProfitPerKmSelection,
            "spatial_synergy": SpatialSynergySelection,
            "stochastic_regret": StochasticRegretSelection,
            "lagrangian": LagrangianSelection,
            "rollout": RolloutSelection,
            "whittle": WhittleIndexSelection,
            "cvar": CVaRSelection,
            "savings": SavingsSelection,
            "set_cover": SetCoverSelection,
            "submodular_greedy": SubmodularGreedySelection,
            "supermodular_greedy": SupermodularGreedySelection,
            "greedy_routing_heuristic": SupermodularGreedySelection,
            "learned": LearnedSelection,
            "wasserstein_robust": WassersteinRobustSelection,
            "dispatcher_thompson": ThompsonDispatcher,
            "dispatcher_portfolio": PortfolioDispatcher,
        }

        # Check explicit registry first
        cls = MandatorySelectionRegistry.get_strategy_class(name)
        if cls:
            # If strategy accepts kwargs, pass them.
            try:
                return cls(**kwargs)
            except TypeError:
                # Fallback for strategies that don't accept args
                return cls()

        # Check default map
        cls_def = cast(Optional[Type[IMandatorySelectionStrategy]], default_map.get(name.lower()))
        if cls_def:
            try:
                return cls_def(**kwargs)
            except TypeError:
                # Fallback for strategies that don't accept args
                return cls_def()

        raise ValueError(f"Unknown selection strategy: {name}")

    @classmethod
    def create_from_config(cls, config: Any) -> IMandatorySelectionStrategy:  # noqa: C901
        """
        Create a selection strategy from a MandatorySelectionConfig object.

        Args:
            config: MandatorySelectionConfig instance.
        """
        if config.strategy is None:
            # Default to all if no strategy specified? Or raise?
            # MandatorySelectionAction currently handles None by skipping.
            raise ValueError("No strategy specified in MandatorySelectionConfig")

        # Map config fields to strategy kwargs
        # This mapping depends on how strategies use their parameters.
        # Based on current implementation, many use 'threshold' from Context,
        # but some might take params in __init__.
        params = config.params.copy()

        # Forward EOQ parameters if present
        for eoq_key in ("use_eoq_threshold", "holding_cost_per_kg_day", "ordering_cost_per_visit"):
            if hasattr(config, eoq_key):
                params[eoq_key] = getattr(config, eoq_key)

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
        elif config.strategy in ("deadline", "multi_day_prob"):
            params.update({"horizon_days": config.horizon_days, "threshold": config.threshold})
        elif config.strategy == "pareto_front":
            params["threshold"] = config.threshold
        elif config.strategy == "profit_per_km":
            params.update(
                {
                    "threshold": config.threshold,
                    "revenue_kg": config.revenue_kg,
                }
            )
        elif config.strategy == "spatial_synergy":
            params.update(
                {
                    "critical_threshold": config.critical_threshold,
                    "synergy_threshold": config.synergy_threshold,
                    "radius": config.radius,
                }
            )
        elif config.strategy == "stochastic_regret":
            params["threshold"] = config.threshold
        elif config.strategy == "lagrangian":
            params.update({"n_vehicles": config.n_vehicles, "cost_per_km": config.cost_per_km})
        elif config.strategy == "rollout":
            params.update(
                {
                    "rollout_horizon": config.rollout_horizon,
                    "rollout_base_policy": config.rollout_base_policy,
                    "rollout_n_scenarios": config.rollout_n_scenarios,
                }
            )
        elif config.strategy == "whittle":
            params.update(
                {
                    "whittle_discount": config.whittle_discount,
                    "whittle_grid_size": config.whittle_grid_size,
                    "n_vehicles": config.n_vehicles,
                }
            )
        elif config.strategy == "cvar":
            params.update({"cvar_alpha": config.cvar_alpha, "threshold": config.threshold})
        elif config.strategy == "savings":
            params["savings_min_fill_ratio"] = config.savings_min_fill_ratio
        elif config.strategy == "set_cover":
            params.update(
                {
                    "service_radius": config.service_radius,
                    "critical_threshold": config.critical_threshold,
                }
            )
        elif config.strategy == "submodular_greedy":
            params.update(
                {
                    "submodular_alpha": config.submodular_alpha,
                    "submodular_budget": config.submodular_budget,
                }
            )
        elif config.strategy == "learned":
            params.update(
                {
                    "learned_model_path": config.learned_model_path,
                    "learned_threshold": config.learned_threshold,
                }
            )
        elif config.strategy == "wasserstein_robust":
            params.update(
                {
                    "wasserstein_radius": config.wasserstein_radius,
                    "wasserstein_p": config.wasserstein_p,
                    "threshold": config.threshold,
                }
            )
        elif config.strategy in ("dispatcher_thompson", "dispatcher_portfolio"):
            params.update(
                {
                    "dispatcher_state_path": config.dispatcher_state_path,
                    "dispatcher_candidate_strategies": config.dispatcher_candidate_strategies,
                    "dispatcher_exploration": config.dispatcher_exploration,
                    "dispatcher_mode": config.dispatcher_mode,
                }
            )

        return cls.create_strategy(config.strategy, **params)
