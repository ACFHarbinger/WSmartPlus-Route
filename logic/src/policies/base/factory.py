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
        import logic.src.policies.route_construction.exact_and_decomposition_solvers as exact_solvers  # noqa

        # Meta-Heuristics
        import logic.src.policies.route_construction.meta_heuristics as meta_heuristics  # noqa

        # Hyper-Heuristics
        import logic.src.policies.route_construction.hyper_heuristics as hyper_heuristics  # noqa

        # Matheuristics (Exact Solvers + Heuristics)
        import logic.src.policies.route_construction.matheuristics as matheuristics  # noqa

        # Learning Algorithms
        import logic.src.policies.route_construction.learning_algorithms as learning_algorithms  # noqa

        # Learning Heuristic Algorithms (Learning Algorithms + Heuristics)
        import logic.src.policies.route_construction.learning_heuristic_algorithms as learning_heuristic_algorithms  # noqa

        # Acceptance Criterion
        import logic.src.policies.route_construction.acceptance_criteria as acceptance_criteria  # noqa

        # Other Algorithms
        import logic.src.policies.route_construction.other_algorithms as other_algorithms  # noqa

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
