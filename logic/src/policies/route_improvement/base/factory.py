"""
Route Improvement Factory Module.

This module implements the Factory pattern for creating route improvement operators.
It handles the instantiation of route improvers based on configuration or names.

Attributes:
    RouteImproverFactory (class): The factory class.

Example:
    >>> from logic.src.policies.helpers.route_improvement.base.factory import RouteImproverFactory
    >>> processors = RouteImproverFactory.create_from_config(config)
"""

from typing import Any, List

from logic.src.interfaces.route_improvement import IRouteImprovement

from .registry import RouteImproverRegistry


class RouteImproverFactory:
    """Factory for creating route improvement strategy instances."""

    @staticmethod
    def create(name: str) -> IRouteImprovement:  # noqa: C901
        """
        Create a route improver instance by name.
        """
        from ..adaptive_large_neighborhood_search import AdaptiveLargeNeighborhoodSearchRouteImprover
        from ..cheapest_insertion import CheapestInsertionRouteImprover
        from ..cross_exchange import CrossExchangeRouteImprover
        from ..fast_tsp import FastTSPRouteImprover
        from ..guided_local_search import GuidedLocalSearchRouteImprover
        from ..lkh import LinKernighanHelsgaunRouteImprover
        from ..local_search import ClassicalLocalSearchRouteImprover
        from ..or_opt import OrOptRouteImprover
        from ..path import PathRouteImprover
        from ..profitable_detour import ProfitableDetourRouteImprover
        from ..random_local_search import RandomLocalSearchRouteImprover
        from ..regret_k_insertion import RegretKInsertionRouteImprover
        from ..ruin_recreate import RuinRecreateRouteImprover
        from ..simulated_annealing import SimulatedAnnealingRouteImprover
        from ..two_phase import TwoPhaseRouteImprover

        cls = RouteImproverRegistry.get_route_improver_class(name)
        if not cls:
            # Fallback for dynamic/mapped names
            n_lower = name.lower()
            if n_lower in ["fast_tsp", "tsp"]:
                return FastTSPRouteImprover()
            elif n_lower in ["local_search", "classical_local_search", "cls"]:
                return ClassicalLocalSearchRouteImprover(operator_name=n_lower)
            elif n_lower in ["random", "random_local_search", "rls"]:
                return RandomLocalSearchRouteImprover()
            elif n_lower in ["lkh", "lkh3", "lkh-3", "lin_kernighan_helsgaun"]:
                return LinKernighanHelsgaunRouteImprover()
            elif n_lower == "path":
                return PathRouteImprover()
            elif n_lower == "or_opt":
                return OrOptRouteImprover()
            elif n_lower == "cross_exchange":
                return CrossExchangeRouteImprover()
            elif n_lower == "guided_local_search":
                return GuidedLocalSearchRouteImprover()
            elif n_lower == "simulated_annealing":
                return SimulatedAnnealingRouteImprover()
            elif n_lower == "cheapest_insertion":
                return CheapestInsertionRouteImprover()
            elif n_lower == "regret_k_insertion":
                return RegretKInsertionRouteImprover()
            elif n_lower == "profitable_detour":
                return ProfitableDetourRouteImprover()
            elif n_lower == "ruin_recreate":
                return RuinRecreateRouteImprover()
            elif n_lower == "adaptive_large_neighborhood_search":
                return AdaptiveLargeNeighborhoodSearchRouteImprover()
            elif n_lower == "two_phase":
                return TwoPhaseRouteImprover()

            raise ValueError(f"Unknown route improver: {name}")
        return cls()

    @classmethod
    def create_from_config(cls, config: Any) -> List[IRouteImprovement]:  # noqa: C901
        """
        Create a list of route improver instances from a RouteImprovingConfig object.

        Args:
            config: RouteImprovingConfig instance.
        """
        from ..adaptive_large_neighborhood_search import AdaptiveLargeNeighborhoodSearchRouteImprover
        from ..cheapest_insertion import CheapestInsertionRouteImprover
        from ..cross_exchange import CrossExchangeRouteImprover
        from ..fast_tsp import FastTSPRouteImprover
        from ..guided_local_search import GuidedLocalSearchRouteImprover
        from ..lkh import LinKernighanHelsgaunRouteImprover
        from ..local_search import ClassicalLocalSearchRouteImprover
        from ..or_opt import OrOptRouteImprover
        from ..path import PathRouteImprover
        from ..profitable_detour import ProfitableDetourRouteImprover
        from ..random_local_search import RandomLocalSearchRouteImprover
        from ..regret_k_insertion import RegretKInsertionRouteImprover
        from ..ruin_recreate import RuinRecreateRouteImprover
        from ..simulated_annealing import SimulatedAnnealingRouteImprover
        from ..two_phase import TwoPhaseRouteImprover

        processors: List[IRouteImprovement] = []
        if not config.methods:
            return processors

        for method in config.methods:
            method_lower = method.lower()
            processor: IRouteImprovement

            # Create route improver with method-specific config parameters
            if method_lower in ["fast_tsp", "tsp"]:
                processor = FastTSPRouteImprover(
                    time_limit=config.fast_tsp.time_limit,
                    seed=config.fast_tsp.seed,
                )
            elif method_lower in ["lkh", "lkh3", "lkh-3", "lin_kernighan_helsgaun"]:
                processor = LinKernighanHelsgaunRouteImprover(
                    max_iterations=config.lkh.max_iterations,
                    time_limit=config.lkh.time_limit,
                    seed=config.lkh.seed,
                )
            elif method_lower in ["local_search", "classical_local_search", "cls"]:
                processor = ClassicalLocalSearchRouteImprover(
                    ls_operator=config.local_search.ls_operator,
                    iterations=config.local_search.iterations,
                    time_limit=config.local_search.time_limit,
                    seed=config.local_search.seed,
                )
            elif method_lower in ["random", "random_local_search", "rls"]:
                processor = RandomLocalSearchRouteImprover(
                    iterations=config.random_local_search.iterations,
                    params=config.random_local_search.params,
                    time_limit=config.random_local_search.time_limit,
                    seed=config.random_local_search.seed,
                )
            elif method_lower == "path":
                processor = PathRouteImprover(
                    vehicle_capacity=config.path.vehicle_capacity,
                )
            elif method_lower == "or_opt":
                processor = OrOptRouteImprover(
                    chain_len=config.or_opt.chain_len,
                    iterations=config.or_opt.iterations,
                    seed=config.or_opt.seed,
                )
            elif method_lower == "cross_exchange":
                processor = CrossExchangeRouteImprover(
                    cross_exchange_max_segment_len=config.cross_exchange.cross_exchange_max_segment_len,
                    iterations=config.cross_exchange.iterations,
                    seed=config.cross_exchange.seed,
                )
            elif method_lower == "guided_local_search":
                processor = GuidedLocalSearchRouteImprover(
                    gls_iterations=config.guided_local_search.gls_iterations,
                    gls_inner_iterations=config.guided_local_search.gls_inner_iterations,
                    gls_lambda_factor=config.guided_local_search.gls_lambda_factor,
                    gls_base_operator=config.guided_local_search.gls_base_operator,
                    seed=config.guided_local_search.seed,
                )
            elif method_lower == "simulated_annealing":
                processor = SimulatedAnnealingRouteImprover(
                    sa_iterations=config.simulated_annealing.sa_iterations,
                    sa_t_init=config.simulated_annealing.sa_t_init,
                    sa_t_min=config.simulated_annealing.sa_t_min,
                    sa_cooling=config.simulated_annealing.sa_cooling,
                    params=config.simulated_annealing.params,
                    seed=config.simulated_annealing.seed,
                )
            elif method_lower == "cheapest_insertion":
                processor = CheapestInsertionRouteImprover(
                    cost_per_km=config.insertion.cost_per_km,
                    revenue_kg=config.insertion.revenue_kg,
                    n_bins=config.insertion.n_bins,
                    seed=config.insertion.seed,
                )
            elif method_lower == "regret_k_insertion":
                processor = RegretKInsertionRouteImprover(
                    regret_k=config.insertion.regret_k,
                    cost_per_km=config.insertion.cost_per_km,
                    revenue_kg=config.insertion.revenue_kg,
                    n_bins=config.insertion.n_bins,
                    seed=config.insertion.seed,
                )
            elif method_lower == "profitable_detour":
                processor = ProfitableDetourRouteImprover(
                    detour_epsilon=config.insertion.detour_epsilon,
                    cost_per_km=config.insertion.cost_per_km,
                    revenue_kg=config.insertion.revenue_kg,
                    n_bins=config.insertion.n_bins,
                    seed=config.insertion.seed,
                )
            elif method_lower == "ruin_recreate":
                processor = RuinRecreateRouteImprover(
                    lns_iterations=config.ruin_recreate.lns_iterations,
                    ruin_fraction=config.ruin_recreate.ruin_fraction,
                    lns_acceptance=config.ruin_recreate.lns_acceptance,
                    lns_sa_temperature=config.ruin_recreate.lns_sa_temperature,
                    cost_per_km=config.ruin_recreate.cost_per_km,
                    revenue_kg=config.ruin_recreate.revenue_kg,
                    seed=config.ruin_recreate.seed,
                )
            elif method_lower == "adaptive_large_neighborhood_search":
                processor = AdaptiveLargeNeighborhoodSearchRouteImprover(
                    alns_iterations=config.adaptive_lns.alns_iterations,
                    ruin_fraction=config.adaptive_lns.ruin_fraction,
                    alns_bandit_warm_start_path=config.adaptive_lns.alns_bandit_warm_start_path,
                    cost_per_km=config.adaptive_lns.cost_per_km,
                    revenue_kg=config.adaptive_lns.revenue_kg,
                    seed=config.adaptive_lns.seed,
                )
            elif method_lower == "two_phase":
                processor = TwoPhaseRouteImprover(
                    phase_one=config.two_phase.phase_one,
                    phase_two=config.two_phase.phase_two,
                    seed=config.two_phase.seed,
                )
            else:
                # Fallback to registry
                processor = cls.create(method)

            processors.append(processor)

        return processors
