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

        # Exact Stochastic and Decomposition Solvers
        import logic.src.policies.exact_and_decomposition_solvers as exact_solvers  # noqa

        # Meta-Heuristics
        import logic.src.policies.meta_heuristics as meta_heuristics  # noqa

        # Hyper-Heuristics
        import logic.src.policies.hyper_heuristics as hyper_heuristics  # noqa

        # Matheuristics (Exact Solvers + Heuristics)
        import logic.src.policies.matheuristics as matheuristics  # noqa

        # Learning Algorithms
        import logic.src.policies.learning_algorithms as learning_algorithms  # noqa

        # Learning Heuristic Algorithms (Learning Algorithms + Heuristics)
        import logic.src.policies.learning_heuristic_algorithms as learning_heuristic_algorithms  # noqa

        # Acceptance Criterion
        import logic.src.policies.boltzmann_metropolis_criterion.policy_bmc as policy_bmc  # noqa
        import logic.src.policies.ensemble_move_acceptance.policy_ema as policy_ema  # noqa
        import logic.src.policies.great_deluge.policy_gd as policy_gd  # noqa
        import logic.src.policies.improving_and_equal.policy_ie as policy_ie  # noqa
        import logic.src.policies.late_acceptance_hill_climbing.policy_lahc as policy_lahc  # noqa
        import logic.src.policies.old_bachelor_acceptance.policy_oba as policy_oba  # noqa
        import logic.src.policies.only_improving.policy_oi as policy_oi  # noqa
        import logic.src.policies.record_to_record_travel.policy_rrt as policy_rrt  # noqa
        import logic.src.policies.step_counting_hill_climbing.policy_schc as policy_schc  # noqa
        import logic.src.policies.threshold_accepting.policy_ta as policy_ta  # noqa

        # Other Algorithms
        import logic.src.policies.other_algorithms as other_algorithms  # noqa

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
