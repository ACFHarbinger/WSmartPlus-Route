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

    @staticmethod
    def get_adapter(
        name: str, engine: Optional[str] = None, threshold: Optional[float] = None, **kwargs: Any
    ) -> IPolicy:
        """
        Create and return the appropriate PolicyAdapter for the given parameters.
        """
        # Local imports to avoid circular dependencies and trigger registration
        import logic.src.policies.neural_agent as neural_agent  # noqa
        import logic.src.policies.adapters.policy_alns as policy_alns  # noqa
        import logic.src.policies.adapters.policy_bcp as policy_bcp  # noqa
        import logic.src.policies.adapters.policy_cvrp as policy_cvrp  # noqa
        import logic.src.policies.adapters.policy_hgs as policy_hgs  # noqa
        import logic.src.policies.adapters.policy_hgs_alns as policy_hgs_alns  # noqa
        import logic.src.policies.adapters.policy_lkh as policy_lkh  # noqa
        import logic.src.policies.adapters.policy_sans as policy_sans  # noqa
        import logic.src.policies.adapters.policy_tsp as policy_tsp  # noqa
        import logic.src.policies.adapters.policy_vrpp as policy_vrpp  # noqa
        import logic.src.policies.adapters.policy_ks_aco as policy_ks_aco  # noqa
        import logic.src.policies.adapters.policy_hh_aco as policy_hh_aco  # noqa
        import logic.src.policies.adapters.policy_sisr as policy_sisr  # noqa

        # Normalize name
        if not isinstance(name, str):
            raise TypeError(f"Policy name must be a string, got {type(name)}")
        name = name.lower()

        # Try Registry first
        cls = PolicyRegistry.get(name) or PolicyRegistry.get(f"policy_{name}")

        if cls:
            return cls()  # type: ignore[return-value]

        raise ValueError(f"Unknown policy: {name}. Ensure it is registered in PolicyRegistry.")


# Backward compatibility aliases
def __getattr__(name: str) -> Any:
    """
    Lazy loader for module-level attributes.
    """
    if name == "NeuralPolicyAdapter":
        from logic.src.policies.adapters.policy_neural import NeuralPolicy

        return NeuralPolicy
    elif name == "VRPPPolicyAdapter":
        from logic.src.policies.adapters.policy_vrpp import VRPPPolicy

        return VRPPPolicy
    elif name == "TSPPolicy":
        from logic.src.policies.adapters.policy_tsp import TSPPolicy

        return TSPPolicy
    elif name == "CVRPPolicy":
        from logic.src.policies.adapters.policy_cvrp import CVRPPolicy

        return CVRPPolicy
    elif name == "ALNSPolicy":
        from logic.src.policies.adapters.policy_alns import ALNSPolicy

        return ALNSPolicy
    elif name == "BCPPolicy":
        from logic.src.policies.adapters.policy_bcp import BCPPolicy

        return BCPPolicy
    elif name == "HGSPolicy":
        from logic.src.policies.adapters.policy_hgs import HGSPolicy

        return HGSPolicy
    elif name == "HGSALNSPolicy":
        from logic.src.policies.adapters.policy_hgs_alns import HGSALNSPolicy

        return HGSALNSPolicy
    elif name == "LKHPolicy":
        from logic.src.policies.adapters.policy_lkh import LKHPolicy

        return LKHPolicy
    elif name == "SANSPolicy":
        from logic.src.policies.adapters.policy_sans import SANSPolicy

        return SANSPolicy
    elif name == "PolicyAdapter":
        return IPolicy
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
