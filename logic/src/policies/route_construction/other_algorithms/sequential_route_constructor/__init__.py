"""Sequential Route Constructor (SRC) package.

This package implements a meta-constructor that executes a fixed chain
of routing heuristics in sequence, threading the state between them.
"""

from .params import SRCParams as SRCParams
from .policy_src import SequentialRouteConstructor as SequentialRouteConstructor

__all__ = ["SequentialRouteConstructor", "SRCParams"]
