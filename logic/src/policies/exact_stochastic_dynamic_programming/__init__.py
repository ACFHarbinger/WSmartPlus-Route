"""
Exact Stochastic Dynamic Programming (SDP) Policy.

Implements Backward Induction to solve small SCWCVRP/SIRP instances optimally.
"""

from .params import SDPParams
from .policy_sdp import exact_sdp_solve
from .sdp_engine import ExactSDPEngine

__all__ = [
    "SDPParams",
    "ExactSDPEngine",
    "exact_sdp_solve",
]
