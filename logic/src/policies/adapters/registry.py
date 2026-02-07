from typing import Callable, Dict, List, Optional, Type

from logic.src.interfaces.policy import IPolicy


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
