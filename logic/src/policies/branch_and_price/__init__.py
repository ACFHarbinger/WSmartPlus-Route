"""
Branch-and-Price Algorithm for VRPP.

This package implements the Branch-and-Price algorithm for the Vehicle Routing
Problem with Profits (VRPP) based on the methodology described in:
Barnhart et al. (1998) "Branch-and-Price: Column Generation for Solving Huge Integer Programs"

The implementation uses:
- Set partitioning master problem formulation
- Column generation via pricing subproblem (resource-constrained shortest path)
- Ryan-Foster branching for binary decisions
- Gurobi for solving restricted master problems

Key Components:
- VRPPMasterProblem: Set partitioning formulation with route columns
- PricingSubproblem: Resource-constrained shortest path for route generation
- BranchAndPriceSolver: Main solver with column generation and branching
- PolicyBP: Policy adapter for WSmart+ Route framework
"""

from .master_problem import VRPPMasterProblem
from .pricing_subproblem import PricingSubproblem
from .rcspp_dp import RCSPPSolver
from .solver import BranchAndPriceSolver
from .policy_bp import PolicyBP, run_branch_and_price

__all__ = [
    "VRPPMasterProblem",
    "PricingSubproblem",
    "RCSPPSolver",
    "BranchAndPriceSolver",
    "PolicyBP",
    "run_branch_and_price",
]
