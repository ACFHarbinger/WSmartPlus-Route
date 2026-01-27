"""
Policy Adapter module - Unified interface for all routing policies.

This module implements the Adapter design pattern to provide a consistent
interface for executing diverse routing policies within the simulator.

Now also includes the IPolicy interface and PolicyRegistry.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type


# --- IPolicy Interface ---
class IPolicy(ABC):
    """
    Interface for all routing policies.
    """

    @abstractmethod
    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the policy to generate a route.

        Args:
            **kwargs: Context dictionary containing simulation state.

        Returns:
            Tuple[List[int], float, Any]: (tour, cost, additional_output)
        """
        pass


# --- Policy Registry ---
class PolicyRegistry:
    """
    Central registry for routing policies.
    """

    _registry: Dict[str, Type[IPolicy]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a policy class.

        Args:
            name: Unique identifier for the policy.
        """

        def decorator(policy_cls: Type[IPolicy]) -> Type[IPolicy]:
            """
            Inner decorator function.

            Args:
                policy_cls: The policy implementation class.
            """
            cls._registry[name] = policy_cls
            return policy_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[IPolicy]]:
        """
        Retrieve a policy class by name.
        """
        return cls._registry.get(name)

    @classmethod
    def list_policies(cls) -> List[str]:
        """
        List all registered policies.
        """
        return list(cls._registry.keys())


# --- Policy Factory ---
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
        # Local imports to avoid circular dependencies
        import logic.src.policies.policy_alns  # noqa
        import logic.src.policies.policy_bcp  # noqa
        import logic.src.policies.policy_cvrp  # noqa
        import logic.src.policies.policy_hgs  # noqa
        import logic.src.policies.policy_lac  # noqa
        import logic.src.policies.policy_lkh  # noqa
        import logic.src.policies.policy_sans  # noqa
        import logic.src.policies.policy_tsp  # noqa
        from logic.src.policies.neural_agent import NeuralPolicy
        from logic.src.policies.policy_vrpp import VRPPPolicy

        # Try Registry first
        cls = PolicyRegistry.get(name) or PolicyRegistry.get(f"policy_{name}")

        if cls:
            return cls()

        # Fallback for complex names or un-registered policies
        if name == "neural" or name[:2] == "am" or name[:4] == "ddam" or "transgcn" in name:
            return NeuralPolicy()
        elif "vrpp" in name:
            return VRPPPolicy()
        elif "sans" in name:
            from logic.src.policies.policy_sans import SANSPolicy

            return SANSPolicy()
        elif "lac" in name:
            from logic.src.policies.policy_lac import LACPolicy

            return LACPolicy()
        elif name == "tsp" or "tsp" in name:
            from logic.src.policies.policy_tsp import TSPPolicy

            return TSPPolicy()
        elif name == "cvrp" or "cvrp" in name:
            from logic.src.policies.policy_cvrp import CVRPPolicy

            return CVRPPolicy()
        else:
            # Default to CVRP or TSP based on vehicles if unknown but looks like a policy
            if "policy" in name:
                from logic.src.policies.policy_cvrp import CVRPPolicy
                from logic.src.policies.policy_tsp import TSPPolicy

                n_vehicles = kwargs.get("n_vehicles", 1)
                return TSPPolicy() if n_vehicles == 1 else CVRPPolicy()

            raise ValueError(f"Unknown policy: {name}")


# Backward compatibility aliases
def __getattr__(name: str) -> Any:
    """
    Lazy loader for module-level attributes.
    """
    if name == "NeuralPolicyAdapter":
        from logic.src.policies.neural_agent import NeuralPolicy

        return NeuralPolicy
    elif name == "VRPPPolicyAdapter":
        from logic.src.policies.policy_vrpp import VRPPPolicy

        return VRPPPolicy
    elif name == "TSPPolicy":
        from logic.src.policies.policy_tsp import TSPPolicy

        return TSPPolicy
    elif name == "CVRPPolicy":
        from logic.src.policies.policy_cvrp import CVRPPolicy

        return CVRPPolicy
    elif name == "ALNSPolicy":
        from logic.src.policies.policy_alns import ALNSPolicy

        return ALNSPolicy
    elif name == "BCPPolicy":
        from logic.src.policies.policy_bcp import BCPPolicy

        return BCPPolicy
    elif name == "HGSPolicy":
        from logic.src.policies.policy_hgs import HGSPolicy

        return HGSPolicy
    elif name == "LKHPolicy":
        from logic.src.policies.policy_lkh import LKHPolicy

        return LKHPolicy
    elif name == "SANSPolicy":
        from logic.src.policies.policy_sans import SANSPolicy

        return SANSPolicy
    elif name == "LACPolicy":
        from logic.src.policies.policy_lac import LACPolicy

        return LACPolicy
    elif name == "PolicyAdapter":
        return IPolicy
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
