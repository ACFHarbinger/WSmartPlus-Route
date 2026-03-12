"""
Policy Adapter module - Unified interface for all routing policies.

This module implements the Adapter design pattern to provide a consistent
interface for executing diverse routing policies within the simulator.

Now also includes the IPolicy interface and PolicyRegistry.
"""

from typing import Any, Optional

# --- IPolicy Interface ---
from logic.src.interfaces.adapter import IPolicyAdapter
from logic.src.policies.adapters.registry import PolicyRegistry

# Alias for backward compatibility
IPolicy = IPolicyAdapter


class PolicyFactory:
    """
    Factory for creating policy adapters.
    """

    _registered = False

    @classmethod
    def ensure_registered(cls) -> None:
        """Import all adapter modules to trigger @PolicyRegistry.register() decorators."""
        if cls._registered:
            return
        import logic.src.policies.ant_colony_optimization.policy_hh_aco as policy_hh_aco  # noqa
        import logic.src.policies.ant_colony_optimization.policy_ks_aco as policy_ks_aco  # noqa
        import logic.src.policies.artificial_bee_colony.policy_abc as policy_abc  # noqa
        import logic.src.policies.augmented_hybrid_volleyball_premier_league.policy_ahvpl as policy_ahvpl  # noqa
        import logic.src.policies.branch_cut_and_price.policy_bcp as policy_bcp  # noqa
        import logic.src.policies.capacitated_vehicle_routing_problem.policy_cvrp as policy_cvrp  # noqa
        import logic.src.policies.continuous_local_search.policy_cls as policy_cls  # noqa
        import logic.src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl as policy_es_mcl  # noqa
        import logic.src.policies.evolution_strategy_mu_kappa_lambda.policy_es_mkl as policy_es_mkl  # noqa
        import logic.src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl as policy_es_mpl  # noqa
        import logic.src.policies.fast_iterative_localized_optimization.policy_filo as policy_filo  # noqa
        import logic.src.policies.firefly_algorithm.policy_fa as policy_fa  # noqa
        import logic.src.policies.genetic_algorithm.policy_ga as policy_ga  # noqa
        import logic.src.policies.genetic_algorithm_memetic_island_model.policy_ga_mim as policy_ga_mim  # noqa
        import logic.src.policies.genetic_algorithm_pure_island_model.policy_ga_pim as policy_ga_pim  # noqa
        import logic.src.policies.genetic_algorithm_stochastic_tournament.policy_ga_st as policy_ga_st  # noqa
        import logic.src.policies.guided_indicators_hyper_heuristic.policy_gihh as policy_gihh  # noqa
        import logic.src.policies.guided_local_search.policy_gls as policy_gls  # noqa
        import logic.src.policies.guided_programming_hyper_heuristic.policy_gphh as policy_gphh  # noqa
        import logic.src.policies.harmony_search.policy_hs as policy_hs  # noqa
        import logic.src.policies.hidden_markov_model_great_deluge.policy_hmm_gd as policy_hmm_gd  # noqa
        import logic.src.policies.hybrid_genetic_search.policy_hgs as policy_hgs  # noqa
        import logic.src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns as policy_hgs_alns  # noqa
        import logic.src.policies.hybrid_genetic_search_ruin_and_recreate.policy_hgsrr as policy_hgsrr  # noqa
        import logic.src.policies.hybrid_iterated_local_search.policy_hils as policy_hils  # noqa
        import logic.src.policies.hybrid_volleyball_premier_league.policy_hvpl as policy_hvpl  # noqa
        import logic.src.policies.hyper_heuristic_us_lk.policy_hulk as policy_hulk  # noqa
        import logic.src.policies.iterated_local_search.policy_ils as policy_ils  # noqa
        import logic.src.policies.knowledge_guided_local_search.policy_kgls as policy_kgls  # noqa
        import logic.src.policies.late_acceptance_hill_climbing.policy_lahc as policy_lahc  # noqa
        import logic.src.policies.league_championship_algorithm.policy_lca as policy_lca  # noqa
        import logic.src.policies.neural_agent.policy_neural as neural_agent  # noqa
        import logic.src.policies.old_bachelor_acceptance.policy_oba as policy_oba  # noqa
        import logic.src.policies.particle_swarm_optimization_distance.policy_psoda as policy_psoda  # noqa
        import logic.src.policies.particle_swarm_optimization_memetic.policy_psoma as policy_psoma  # noqa
        import logic.src.policies.quantum_differential_evolution.policy_qde as policy_qde  # noqa
        import logic.src.policies.reactive_tabu_search.policy_rts as policy_rts  # noqa
        import logic.src.policies.record_to_record_travel.policy_rrt as policy_rrt  # noqa
        import logic.src.policies.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns as policy_rl_alns  # noqa
        import logic.src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl as policy_rl_ahvpl  # noqa
        import logic.src.policies.reinforcement_learning_hybrid_volleyball_premier_league.policy_rl_hvpl as policy_rl_hvpl  # noqa
        import logic.src.policies.simulated_annealing.policy_sa as policy_sa  # noqa
        import logic.src.policies.simulated_annealing_neighborhood_search.policy_sans as policy_sans  # noqa
        import logic.src.policies.sine_cosine_algorithm.policy_sca as policy_sca  # noqa
        import logic.src.policies.slack_induction_by_string_removal.policy_sisr as policy_sisr  # noqa
        import logic.src.policies.soccer_league_competition.policy_slc as policy_slc  # noqa
        import logic.src.policies.travelling_salesman_problem.policy_tsp as policy_tsp  # noqa
        import logic.src.policies.variable_neighborhood_search.policy_vns as policy_vns  # noqa
        import logic.src.policies.vehicle_routing_problem_with_profits.policy_vrpp as policy_vrpp  # noqa

        cls._registered = True

    @staticmethod
    def get_adapter(
        name: str,
        config: Optional[dict] = None,
        engine: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> IPolicy:
        """
        Create and return the appropriate PolicyAdapter for the given parameters.

        Args:
            name: Policy name (e.g., 'alns', 'hgs', 'tsp').
            config: Raw policy config dict from YAML. If provided, the adapter's
                    typed config dataclass is built automatically.
            engine: Deprecated. Engine should be specified in config.
            threshold: Deprecated. Threshold should be specified in config.
            **kwargs: Additional keyword arguments (unused, for backward compat).

        Returns:
            Instantiated policy adapter with typed config.
        """
        PolicyFactory.ensure_registered()

        # Normalize name
        if not isinstance(name, str):
            raise TypeError(f"Policy name must be a string, got {type(name)}")
        name = name.lower()

        # Try Registry first
        cls = PolicyRegistry.get(name) or PolicyRegistry.get(f"policy_{name}")

        if cls:
            if config is not None:
                return cls(config=config)  # type: ignore[return-value,call-arg]
            return cls()  # type: ignore[return-value]

        raise ValueError(f"Unknown policy: {name}. Ensure it is registered in PolicyRegistry.")
