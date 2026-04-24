"""Route Improvement Base Package.

This package defines the core infrastructure for the "Route Improvement"
policy, including the abstract interfaces, the factory pattern for algorithm
creation, and the registry for available algorithms.

Attributes:
    IRouteImprovement (Type[IRouteImprovement]): Interface for route improvement algorithms.
    RouteImproverFactory (Type[RouteImproverFactory]): Factory for creating algorithms.
    RouteImproverRegistry (Type[RouteImproverRegistry]): Registry for algorithm classes.

Example:
    >>> from logic.src.policies.route_improvement.base import RouteImproverFactory
    >>> factory = RouteImproverFactory()
    >>> algorithm = factory.get_algorithm("fast_tsp", config=cfg)
"""

from logic.src.interfaces import IRouteImprovement

from .factory import RouteImproverFactory
from .registry import RouteImproverRegistry

__all__ = [
    "IRouteImprovement",
    "RouteImproverRegistry",
    "RouteImproverFactory",
]
