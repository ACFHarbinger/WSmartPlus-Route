"""Registry for joint selection and construction policies."""

from typing import Any, Dict, Optional, Type


class JointPolicyRegistry:
    """Registry for joint selection and construction policies.

    Maintains a mapping of policy names to their respective classes, enabling
    dynamic instantiation through the factory.

    Attributes:
        _registry (Dict[str, Type]): Internal mapping of policy names to class types.
    """

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """Register a joint selection and construction policy class.

        Args:
            name (str): Unique name identifier for the policy.

        Returns:
            Callable: A decorator that registers the subclass.
        """
        def decorator(subclass: Type[Any]):
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Retrieve a registered policy class by name.

        Args:
            name (str): Unique name identifier for the policy.

        Returns:
            Optional[Type]: The policy class if found, else None.
        """
        return cls._registry.get(name.lower())

    @classmethod
    def list_policies(cls) -> list[str]:
        """List all registered policy names.

        Returns:
            list[str]: Sorted list of registered policy names.
        """
        return sorted(cls._registry.keys())
