"""
Post-Processing Factory Module.

This module implements the Factory pattern for creating post-processing operators.
It handles the instantiation of post-processors based on configuration or names.

Attributes:
    PostProcessorFactory (class): The factory class.

Example:
    >>> from logic.src.policies.other.post_processing.base.factory import PostProcessorFactory
    >>> processors = PostProcessorFactory.create_from_config(config)
"""

from typing import Any, List

from logic.src.interfaces.post_processing import IPostProcessor

from .registry import PostProcessorRegistry


class PostProcessorFactory:
    """Factory for creating post-processing strategy instances."""

    @staticmethod
    def create(name: str) -> IPostProcessor:  # noqa: C901
        """
        Create a post-processor instance by name.
        """
        from ..adaptive_large_neighborhood_search import AdaptiveLargeNeighborhoodSearchPostProcessor
        from ..cheapest_insertion import CheapestInsertionPostProcessor
        from ..cross_exchange import CrossExchangePostProcessor
        from ..fast_tsp import FastTSPPostProcessor
        from ..guided_local_search import GuidedLocalSearchPostProcessor
        from ..lkh import LinKernighanHelsgaunPostProcessor
        from ..local_search import ClassicalLocalSearchPostProcessor
        from ..or_opt import OrOptPostProcessor
        from ..path import PathPostProcessor
        from ..profitable_detour import ProfitableDetourPostProcessor
        from ..random_local_search import RandomLocalSearchPostProcessor
        from ..regret_k_insertion import RegretKInsertionPostProcessor
        from ..ruin_recreate import RuinRecreatePostProcessor
        from ..simulated_annealing import SimulatedAnnealingPostProcessor
        from ..two_phase import TwoPhasePostProcessor

        cls = PostProcessorRegistry.get_post_processor_class(name)
        if not cls:
            # Fallback for dynamic/mapped names
            n_lower = name.lower()
            if n_lower in ["fast_tsp", "tsp"]:
                return FastTSPPostProcessor()
            elif n_lower in ["local_search", "classical_local_search", "cls"]:
                return ClassicalLocalSearchPostProcessor(operator_name=n_lower)
            elif n_lower in ["random", "random_local_search", "rls"]:
                return RandomLocalSearchPostProcessor()
            elif n_lower in ["lkh", "lkh3", "lkh-3", "lin_kernighan_helsgaun"]:
                return LinKernighanHelsgaunPostProcessor()
            elif n_lower == "path":
                return PathPostProcessor()
            elif n_lower == "or_opt":
                return OrOptPostProcessor()
            elif n_lower == "cross_exchange":
                return CrossExchangePostProcessor()
            elif n_lower == "guided_local_search":
                return GuidedLocalSearchPostProcessor()
            elif n_lower == "simulated_annealing":
                return SimulatedAnnealingPostProcessor()
            elif n_lower == "cheapest_insertion":
                return CheapestInsertionPostProcessor()
            elif n_lower == "regret_k_insertion":
                return RegretKInsertionPostProcessor()
            elif n_lower == "profitable_detour":
                return ProfitableDetourPostProcessor()
            elif n_lower == "ruin_recreate":
                return RuinRecreatePostProcessor()
            elif n_lower == "adaptive_large_neighborhood_search":
                return AdaptiveLargeNeighborhoodSearchPostProcessor()
            elif n_lower == "two_phase":
                return TwoPhasePostProcessor()

            raise ValueError(f"Unknown post-processor: {name}")
        return cls()

    @classmethod
    def create_from_config(cls, config: Any) -> List[IPostProcessor]:  # noqa: C901
        """
        Create a list of post-processor instances from a PostProcessingConfig object.

        Args:
            config: PostProcessingConfig instance.
        """
        from ..adaptive_large_neighborhood_search import AdaptiveLargeNeighborhoodSearchPostProcessor
        from ..cheapest_insertion import CheapestInsertionPostProcessor
        from ..cross_exchange import CrossExchangePostProcessor
        from ..fast_tsp import FastTSPPostProcessor
        from ..guided_local_search import GuidedLocalSearchPostProcessor
        from ..lkh import LinKernighanHelsgaunPostProcessor
        from ..local_search import ClassicalLocalSearchPostProcessor
        from ..or_opt import OrOptPostProcessor
        from ..path import PathPostProcessor
        from ..profitable_detour import ProfitableDetourPostProcessor
        from ..random_local_search import RandomLocalSearchPostProcessor
        from ..regret_k_insertion import RegretKInsertionPostProcessor
        from ..ruin_recreate import RuinRecreatePostProcessor
        from ..simulated_annealing import SimulatedAnnealingPostProcessor
        from ..two_phase import TwoPhasePostProcessor

        processors: List[IPostProcessor] = []
        if not config.methods:
            return processors

        for method in config.methods:
            method_lower = method.lower()
            processor: IPostProcessor

            # Create post-processor with method-specific config parameters
            if method_lower in ["fast_tsp", "tsp"]:
                processor = FastTSPPostProcessor(
                    time_limit=config.fast_tsp.time_limit,
                    seed=config.fast_tsp.seed,
                )
            elif method_lower in ["lkh", "lkh3", "lkh-3", "lin_kernighan_helsgaun"]:
                processor = LinKernighanHelsgaunPostProcessor(
                    max_iterations=config.lkh.max_iterations,
                    time_limit=config.lkh.time_limit,
                    seed=config.lkh.seed,
                )
            elif method_lower in ["local_search", "classical_local_search", "cls"]:
                processor = ClassicalLocalSearchPostProcessor(
                    ls_operator=config.local_search.ls_operator,
                    iterations=config.local_search.iterations,
                    time_limit=config.local_search.time_limit,
                    seed=config.local_search.seed,
                )
            elif method_lower in ["random", "random_local_search", "rls"]:
                processor = RandomLocalSearchPostProcessor(
                    iterations=config.random_local_search.iterations,
                    params=config.random_local_search.params,
                    time_limit=config.random_local_search.time_limit,
                    seed=config.random_local_search.seed,
                )
            elif method_lower == "path":
                processor = PathPostProcessor(
                    vehicle_capacity=config.path.vehicle_capacity,
                )
            elif method_lower == "or_opt":
                processor = OrOptPostProcessor(
                    chain_len=config.or_opt.chain_len,
                    iterations=config.or_opt.iterations,
                    seed=config.or_opt.seed,
                )
            elif method_lower == "cross_exchange":
                processor = CrossExchangePostProcessor(
                    cross_exchange_max_segment_len=config.cross_exchange.cross_exchange_max_segment_len,
                    iterations=config.cross_exchange.iterations,
                    seed=config.cross_exchange.seed,
                )
            elif method_lower == "guided_local_search":
                processor = GuidedLocalSearchPostProcessor(
                    gls_iterations=config.guided_local_search.gls_iterations,
                    gls_inner_iterations=config.guided_local_search.gls_inner_iterations,
                    gls_lambda_factor=config.guided_local_search.gls_lambda_factor,
                    gls_base_operator=config.guided_local_search.gls_base_operator,
                    seed=config.guided_local_search.seed,
                )
            elif method_lower == "simulated_annealing":
                processor = SimulatedAnnealingPostProcessor(
                    sa_iterations=config.simulated_annealing.sa_iterations,
                    sa_t_init=config.simulated_annealing.sa_t_init,
                    sa_t_min=config.simulated_annealing.sa_t_min,
                    sa_cooling=config.simulated_annealing.sa_cooling,
                    params=config.simulated_annealing.params,
                    seed=config.simulated_annealing.seed,
                )
            elif method_lower == "cheapest_insertion":
                processor = CheapestInsertionPostProcessor(
                    cost_per_km=config.insertion.cost_per_km,
                    revenue_kg=config.insertion.revenue_kg,
                    n_bins=config.insertion.n_bins,
                    seed=config.insertion.seed,
                )
            elif method_lower == "regret_k_insertion":
                processor = RegretKInsertionPostProcessor(
                    regret_k=config.insertion.regret_k,
                    cost_per_km=config.insertion.cost_per_km,
                    revenue_kg=config.insertion.revenue_kg,
                    n_bins=config.insertion.n_bins,
                    seed=config.insertion.seed,
                )
            elif method_lower == "profitable_detour":
                processor = ProfitableDetourPostProcessor(
                    detour_epsilon=config.insertion.detour_epsilon,
                    cost_per_km=config.insertion.cost_per_km,
                    revenue_kg=config.insertion.revenue_kg,
                    n_bins=config.insertion.n_bins,
                    seed=config.insertion.seed,
                )
            elif method_lower == "ruin_recreate":
                processor = RuinRecreatePostProcessor(
                    lns_iterations=config.ruin_recreate.lns_iterations,
                    ruin_fraction=config.ruin_recreate.ruin_fraction,
                    lns_acceptance=config.ruin_recreate.lns_acceptance,
                    lns_sa_temperature=config.ruin_recreate.lns_sa_temperature,
                    cost_per_km=config.ruin_recreate.cost_per_km,
                    revenue_kg=config.ruin_recreate.revenue_kg,
                    seed=config.ruin_recreate.seed,
                )
            elif method_lower == "adaptive_large_neighborhood_search":
                processor = AdaptiveLargeNeighborhoodSearchPostProcessor(
                    alns_iterations=config.adaptive_lns.alns_iterations,
                    ruin_fraction=config.adaptive_lns.ruin_fraction,
                    alns_bandit_warm_start_path=config.adaptive_lns.alns_bandit_warm_start_path,
                    cost_per_km=config.adaptive_lns.cost_per_km,
                    revenue_kg=config.adaptive_lns.revenue_kg,
                    seed=config.adaptive_lns.seed,
                )
            elif method_lower == "two_phase":
                processor = TwoPhasePostProcessor(
                    phase_one=config.two_phase.phase_one,
                    phase_two=config.two_phase.phase_two,
                    seed=config.two_phase.seed,
                )
            else:
                # Fallback to registry
                processor = cls.create(method)

            processors.append(processor)

        return processors
