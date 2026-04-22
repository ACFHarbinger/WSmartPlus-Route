"""
Tag registry for WSmart-Route.

Attributes:
    AnyTag: Union of all tags
    T_Algorithm: Type alias for algorithm
    T: TypeVar for type hinting
    GlobalRegistry: Class for registering and querying algorithms

Example:
    >>> from logic.src.enums import GlobalRegistry
    >>> GlobalRegistry.register(PolicyTag.EXACT, ModelTag.TRANSFORMER)
    >>> GlobalRegistry.query_intersection(PolicyTag.EXACT, ModelTag.TRANSFORMER)
    [<class 'logic.src.policies.exact.exact_policy.ExactPolicy'>]
"""

from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

from .environment_tags import EnvironmentTag
from .model_tags import ModelTag
from .operator_tags import OperatorTag
from .policy_tags import PolicyTag
from .trainer_tags import TrainerTag

# The Unified Type Alias
AnyTag = Union[PolicyTag, ModelTag, EnvironmentTag, OperatorTag, TrainerTag]

# Type alias for clarity: can be a Class Type or a Function
T_Algorithm = Union[Type, Callable[..., Any]]
T = TypeVar("T")


class GlobalRegistry:
    """
    Class for registering and querying algorithms.

    Attributes:
        _registry: Dictionary for storing registered algorithms

    Methods:
        register: Register an algorithm with tags
        get_name: Get the name of an algorithm
        query_intersection: Query algorithms matching all tags
        query_union: Query algorithms matching at least one tag
        query_difference: Query algorithms matching required tags but not excluded tags
        get_all: Get all registered algorithms
    """

    _registry: Dict[T_Algorithm, Set[AnyTag]] = {}

    @classmethod
    def register(cls, *tags: AnyTag) -> Callable[[T_Algorithm], T_Algorithm]:
        """
        Universal decorator for any class or function.

        Args:
            *tags: Tags to register the algorithm with

        Returns:
            Decorator for registering the algorithm
        """

        def decorator(obj: T_Algorithm) -> T_Algorithm:
            cls._registry[obj] = set(tags)
            return obj

        return decorator

    @classmethod
    def get_name(cls, obj: T_Algorithm) -> str:
        """
        Helper to get a readable name for either a class or a function.

        Args:
            obj: Algorithm to get the name of

        Returns:
            Name of the algorithm
        """
        return getattr(obj, "__name__", str(obj))

    @classmethod
    def query_intersection(
        cls, *tags: AnyTag, expected_type: Optional[Type[T]] = None
    ) -> Union[List[T_Algorithm], List[Type[T]]]:
        """
        Returns objects matching ALL tags.
        If expected_type is provided, strictly filters out non-matching classes.

        Args:
            *tags: Tags to query for
            expected_type: Expected type of the algorithm

        Returns:
            List of algorithms matching all tags
        """
        query_set = set(tags)
        results: List[Any] = []

        for obj, obj_tags in cls._registry.items():
            if query_set.issubset(obj_tags):
                # If the user only wants specific Base Classes (e.g., nn.Module)
                if expected_type is not None:
                    if isinstance(obj, type) and issubclass(obj, expected_type):
                        results.append(obj)
                else:
                    results.append(obj)

        return results

    @classmethod
    def query_union(
        cls, *tags: AnyTag, expected_type: Optional[Type[T]] = None
    ) -> Union[List[T_Algorithm], List[Type[T]]]:
        """
        Returns objects matching AT LEAST ONE tag.
        If expected_type is provided, strictly filters out non-matching classes.

        Args:
            *tags: Tags to query for
            expected_type: Expected type of the algorithm

        Returns:
            List of algorithms matching at least one tag
        """
        query_set = set(tags)
        results: List[Any] = []

        for obj, obj_tags in cls._registry.items():
            if not query_set.isdisjoint(obj_tags):
                # If the user only wants specific Base Classes (e.g., nn.Module)
                if expected_type is not None:
                    if isinstance(obj, type) and issubclass(obj, expected_type):
                        results.append(obj)
                else:
                    results.append(obj)

        return results

    @classmethod
    def query_difference(
        cls, require: List[AnyTag], exclude: List[AnyTag], expected_type: Optional[Type[T]] = None
    ) -> Union[List[T_Algorithm], List[Type[T]]]:
        """
        Returns algorithms that contain ALL required tags, but NO excluded tags.

        Args:
            require: Tags that must be present
            exclude: Tags that must not be present
            expected_type: Expected type of the algorithm

        Returns:
            List of algorithms matching the criteria
        """
        require_set = set(require)
        exclude_set = set(exclude)
        results: List[Any] = []
        for obj, obj_tags in cls._registry.items():
            if require_set.issubset(obj_tags) and exclude_set.isdisjoint(obj_tags):
                # If the user only wants specific Base Classes (e.g., nn.Module)
                if expected_type is not None:
                    if isinstance(obj, type) and issubclass(obj, expected_type):
                        results.append(obj)
                else:
                    results.append(obj)

        return results

    @classmethod
    def get_all(cls) -> Dict[T_Algorithm, Set[AnyTag]]:
        """
        Get all registered algorithms.

        Returns:
            Dictionary of all registered algorithms
        """
        return cls._registry
