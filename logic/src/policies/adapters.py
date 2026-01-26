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

    Implements the Factory Method pattern to instantiate the appropriate
    PolicyAdapter based on the policy name string.

    Policy Name Patterns:
    ---------------------
    - 'policy_regular*': RegularPolicyAdapter
    - 'policy_last_minute*': LastMinutePolicyAdapter
    - 'am*', 'ddam*', 'transgcn*': NeuralPolicyAdapter
    - '*vrpp*' with 'gurobi' or 'hexaly': VRPPPolicyAdapter
    - 'policy_look_ahead*': LookAheadPolicyAdapter
    """

    @staticmethod
    def get_adapter(policy_name: str) -> IPolicy:
        """
        Create and return the appropriate PolicyAdapter for the given policy name.

        Args:
            policy_name (str): Policy identifier string

        Returns:
            IPolicy: Concrete adapter instance

        Raises:
            ValueError: If policy name doesn't match any known pattern
        """
        # Local imports to avoid circular dependencies
        # as these modules import IPolicy/PolicyRegistry from this file
        from logic.src.policies.last_minute import LastMinutePolicy, ProfitPolicy
        from logic.src.policies.look_ahead import LookAheadPolicy
        from logic.src.policies.neural_agent import NeuralPolicy
        from logic.src.policies.policy_vrpp import VRPPPolicy
        from logic.src.policies.regular import RegularPolicy

        # Try Registry first
        cls = PolicyRegistry.get(policy_name)
        if cls:
            return cls()

        if "policy_last_minute" in policy_name:
            return LastMinutePolicy()
        elif "policy_regular" in policy_name:
            return RegularPolicy()
        elif policy_name[:2] == "am" or policy_name[:4] == "ddam" or "transgcn" in policy_name:
            return NeuralPolicy()
        elif ("gurobi" in policy_name or "hexaly" in policy_name) and "vrpp" in policy_name:
            return VRPPPolicy()
        elif "policy_profit" in policy_name:
            return ProfitPolicy()
        elif "policy_look_ahead" in policy_name:
            return LookAheadPolicy()
        else:
            raise ValueError(f"Unknown policy: {policy_name}")


# Backward compatibility aliases
# We use __getattr__ to lazily import these classes to avoid circular dependencies
# since these modules import IPolicy from this file.


def __getattr__(name: str) -> Any:
    """
    Lazy loader for module-level attributes (backward compatibility aliases).
    """
    if name == "LastMinutePolicyAdapter":
        from logic.src.policies.last_minute import LastMinutePolicy

        return LastMinutePolicy
    elif name == "ProfitPolicyAdapter":
        from logic.src.policies.last_minute import ProfitPolicy

        return ProfitPolicy
    elif name == "NeuralPolicyAdapter":
        from logic.src.policies.neural_agent import NeuralPolicy

        return NeuralPolicy
    elif name == "VRPPPolicyAdapter":
        from logic.src.policies.policy_vrpp import VRPPPolicy

        return VRPPPolicy
    elif name == "RegularPolicyAdapter":
        from logic.src.policies.regular import RegularPolicy

        return RegularPolicy
    elif name == "LookAheadPolicyAdapter":
        from logic.src.policies.look_ahead import LookAheadPolicy

        return LookAheadPolicy
    elif name == "PolicyAdapter":
        return IPolicy
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
