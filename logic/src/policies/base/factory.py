"""
Policy Adapter module - Unified interface for all routing policies.

This module implements the Adapter design pattern to provide a consistent
interface for executing diverse routing policies within the simulator.

Now also includes the IPolicy interface and PolicyRegistry.
"""

from typing import Any, Optional

# --- IPolicy Interface ---
from logic.src.interfaces.adapter import IPolicyAdapter

from .registry import PolicyRegistry

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

        # Exact Solvers
        import logic.src.policies.branch_and_bound.policy_bb as policy_bb  # noqa
        import logic.src.policies.branch_and_price_and_cut.policy_bpc as policy_bpc  # noqa
        import logic.src.policies.vehicle_routing_problem_with_profits.policy_vrpp as policy_vrpp  # noqa

        # Meta-Heuristics
        import logic.src.policies.adaptive_large_neighborhood_search.policy_alns as policy_alns  # noqa
        import logic.src.policies.ant_colony_optimization_k_sparse.policy_ks_aco as policy_ks_aco  # noqa
        import logic.src.policies.artificial_bee_colony.policy_abc as policy_abc  # noqa
        import logic.src.policies.augmented_hybrid_volleyball_premier_league.policy_ahvpl as policy_ahvpl  # noqa
        import logic.src.policies.differential_evolution.policy_de as policy_de  # noqa
        import logic.src.policies.evolution_strategy_mu_comma_lambda.policy_es_mcl as policy_es_mcl  # noqa
        import logic.src.policies.evolution_strategy_mu_kappa_lambda.policy_es_mkl as policy_es_mkl  # noqa
        import logic.src.policies.evolution_strategy_mu_plus_lambda.policy_es_mpl as policy_es_mpl  # noqa
        import logic.src.policies.fast_iterative_localized_optimization.policy_filo as policy_filo  # noqa
        import logic.src.policies.firefly_algorithm.policy_fa as policy_fa  # noqa
        import logic.src.policies.genetic_algorithm.policy_ga as policy_ga  # noqa
        import logic.src.policies.guided_local_search.policy_gls as policy_gls  # noqa
        import logic.src.policies.harmony_search.policy_hs as policy_hs  # noqa
        import logic.src.policies.hybrid_genetic_search.policy_hgs as policy_hgs  # noqa
        import logic.src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns as policy_hgs_alns  # noqa
        import logic.src.policies.hybrid_genetic_search_ruin_and_recreate.policy_hgsrr as policy_hgsrr  # noqa
        import logic.src.policies.hybrid_memetic_search.policy_hms as policy_hms  # noqa
        import logic.src.policies.hybrid_volleyball_premier_league.policy_hvpl as policy_hvpl  # noqa
        import logic.src.policies.iterated_local_search.policy_ils as policy_ils  # noqa
        import logic.src.policies.knowledge_guided_local_search.policy_kgls as policy_kgls  # noqa
        import logic.src.policies.league_championship_algorithm.policy_lca as policy_lca  # noqa
        import logic.src.policies.memetic_algorithm.policy_ma as policy_ma  # noqa
        import logic.src.policies.memetic_algorithm_dual_population.policy_ma_dp as policy_ma_dp  # noqa
        import logic.src.policies.memetic_algorithm_island_model.policy_ma_im as policy_ma_im  # noqa
        import logic.src.policies.memetic_algorithm_tolerance_based_selection.policy_ma_ts as policy_ma_ts  # noqa
        import logic.src.policies.particle_swarm_optimization.policy_pso as policy_pso  # noqa
        import logic.src.policies.particle_swarm_optimization_distance_based_algorithm.policy_psoda as policy_psoda  # noqa
        import logic.src.policies.particle_swarm_optimization_memetic_algorithm.policy_psoma as policy_psoma  # noqa
        import logic.src.policies.quantum_differential_evolution.policy_qde as policy_qde  # noqa
        import logic.src.policies.reactive_tabu_search.policy_rts as policy_rts  # noqa
        import logic.src.policies.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns as policy_rl_alns  # noqa
        import logic.src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.policy_rl_ahvpl as policy_rl_ahvpl  # noqa
        import logic.src.policies.reinforcement_learning_hybrid_volleyball_premier_league.policy_rl_hvpl as policy_rl_hvpl  # noqa
        import logic.src.policies.simulated_annealing_neighborhood_search.policy_sans as policy_sans  # noqa
        import logic.src.policies.sine_cosine_algorithm.policy_sca as policy_sca  # noqa
        import logic.src.policies.slack_induction_by_string_removal.policy_sisr as policy_sisr  # noqa
        import logic.src.policies.soccer_league_competition.policy_slc as policy_slc  # noqa
        import logic.src.policies.variable_neighborhood_search.policy_vns as policy_vns  # noqa
        import logic.src.policies.volleyball_premier_league.policy_vpl as policy_vpl  # noqa

        # Hyper-Heuristics
        import logic.src.policies.ant_colony_optimization_hyper_heuristic.policy_hh_aco as policy_hh_aco  # noqa
        import logic.src.policies.genetic_programming_hyper_heuristic.policy_gphh as policy_gphh  # noqa
        import logic.src.policies.guided_indicators_hyper_heuristic.policy_gihh as policy_gihh  # noqa
        import logic.src.policies.hidden_markov_model_great_deluge_hyper_heuristic.policy_hmm_gd_hh as policy_hmm_gd_hh  # noqa
        import logic.src.policies.hyper_heuristic_us_lk.policy_hulk as policy_hulk  # noqa
        import logic.src.policies.reinforcement_learning_great_deluge_hyper_heuristic.policy_rl_gd_hh as policy_rl_gd_hh  # noqa
        import logic.src.policies.sequence_based_selection_hyper_heuristic.policy_ss_hh as policy_ss_hh  # noqa

        # Matheuristics
        import logic.src.policies.adaptive_kernel_search.policy_aks as policy_aks  # noqa
        import logic.src.policies.cluster_first_route_second.policy_cf_rs as policy_cf_rs  # noqa
        import logic.src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp as policy_ils_rvnd_sp  # noqa
        import logic.src.policies.kernel_search.policy_ks as policy_ks  # noqa
        import logic.src.policies.popmusic.policy_popmusic as policy_popmusic  # noqa
        import logic.src.policies.rens.policy_rens as policy_rens  # noqa

        # Acceptance Criterion
        import logic.src.policies.ensemble_move_acceptance.policy_ema as policy_ema  # noqa
        import logic.src.policies.great_deluge.policy_gd as policy_gd  # noqa
        import logic.src.policies.improving_and_equal.policy_ie as policy_ie  # noqa
        import logic.src.policies.late_acceptance_hill_climbing.policy_lahc as policy_lahc  # noqa
        import logic.src.policies.old_bachelor_acceptance.policy_oba as policy_oba  # noqa
        import logic.src.policies.only_improving.policy_oi as policy_oi  # noqa
        import logic.src.policies.record_to_record_travel.policy_rrt as policy_rrt  # noqa
        import logic.src.policies.simulated_annealing.policy_sa as policy_sa  # noqa
        import logic.src.policies.threshold_accepting.policy_ta as policy_ta  # noqa

        # Others
        import logic.src.policies.capacitated_vehicle_routing_problem.policy_cvrp as policy_cvrp  # noqa
        import logic.src.policies.travelling_salesman_problem.policy_tsp as policy_tsp  # noqa
        import logic.src.policies.neural_agent.policy_neural as neural_agent  # noqa

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
