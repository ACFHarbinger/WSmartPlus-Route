"""Sequential Route Constructor (SRC) package.

This package implements a meta-constructor that executes a fixed chain
of routing heuristics in sequence, threading the state between them.

Attributes:
    SRCParams: Parameters for the Sequential Route Constructor.
    SequentialRouteConstructor: Sequential Route Constructor (SRC).

Example:
    >>> from logic.src.policies.route_construction.other_algorithms.sequential_route_constructor import SequentialRouteConstructor, SRCParams
    >>> params = SRCParams()
    >>> src = SequentialRouteConstructor(config=params)
    >>> src.execute(mandatory=[1, 2, 3], bins=bins_object, distance_matrix=distance_matrix)
"""

from .params import SRCParams as SRCParams
from .policy_src import SequentialRouteConstructor as SequentialRouteConstructor

__all__ = ["SequentialRouteConstructor", "SRCParams"]
