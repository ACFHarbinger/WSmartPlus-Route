"""
Route Constructor Base Package.

This package provides adapter implementations for various routing
optimization policies (HGS, ALNS, VRPP, etc.) and a factory pattern
for dynamic route constructor instantiation.

Attributes:
    BaseRoutingPolicy: Template base class for all routing policies.
    BaseMultiPeriodRoutingPolicy: Base class for multi-period (T-day) routing.
    IRouteConstructor: Interface defining the route construction contract.
    RouteConstructorFactory: Factory for instantiating registered constructors.
    RouteConstructorRegistry: Registry for dynamic constructor lookup.

Example:
    >>> from logic.src.policies.route_construction.base import RouteConstructorFactory
    >>> adapter = RouteConstructorFactory.get_adapter("hgs")
    >>> tour, cost, profit, _, _ = adapter.execute(mandatory=[1, 2, 3], ...)

"""

from .base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from .base_routing_policy import BaseRoutingPolicy
from .factory import IRouteConstructor, RouteConstructorFactory, RouteConstructorRegistry

__all__ = [
    "BaseRoutingPolicy",
    "BaseMultiPeriodRoutingPolicy",
    "IRouteConstructor",
    "RouteConstructorFactory",
    "RouteConstructorRegistry",
]
