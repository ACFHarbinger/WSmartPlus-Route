"""Adaptive Route Constructor Orchestrator (ARCO) package.

This package implements an adaptive meta-constructor that learns optimal
ordering of routing heuristics through online experience.

Attributes:
    ARCOParams: Parameters for the ARCO algorithm.
    AdaptiveRouteConstructorOrchestrator: The ARCO algorithm itself.

Example:
    >>> from logic.src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator import ARCOParams, AdaptiveRouteConstructorOrchestrator
    >>> params = ARCOParams()
    >>> arco = AdaptiveRouteConstructorOrchestrator(params)
    >>> routes = arco.build_routes(context)
"""

from .params import ARCOParams as ARCOParams
from .policy_arco import AdaptiveRouteConstructorOrchestrator as AdaptiveRouteConstructorOrchestrator

__all__ = ["AdaptiveRouteConstructorOrchestrator", "ARCOParams"]
