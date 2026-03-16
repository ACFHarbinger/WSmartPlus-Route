"""
Branch-and-Bound Policy Module.

This package implements the foundational Branch-and-Bound (BB) algorithm for exact
combinatorial optimization, specifically specialized for the Vehicle Routing
Problem with Profits (VRPP) and Capacitated Waste Collection VRP (CWC VRP).

Implementation follows the methodology proposed by Land and Doig (1960), utilizing
Linear Programming (LP) relaxations to provide rigorous mathematical bounds during
tree exploration.
"""

from .policy_bb import BranchAndBoundPolicy
from .solver import run_bb

__all__ = ["BranchAndBoundPolicy", "run_bb"]
