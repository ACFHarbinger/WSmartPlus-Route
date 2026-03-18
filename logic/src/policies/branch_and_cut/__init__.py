"""
Branch-and-Cut Algorithm for Vehicle Routing Problem with Profits (VRPP).

This module implements an exact branch-and-cut algorithm for VRPP based on
cutting plane methods from:
- Fischetti, M., Salazar González, J. J., & Toth, P. (1997).
  "A Branch-And-Cut Algorithm for the Symmetric Generalized Traveling Salesman Problem"
- Padberg, M., & Rinaldi, G. (1991). "A Branch-and-cut Algorithm for the Resolution
  of Large-scale Symmetric Traveling Salesman Problems"

The VRPP combines node selection (which nodes to visit) with routing optimization,
aiming to maximize profit (waste collected) minus travel cost.
"""

from logic.src.policies.branch_and_cut.bc import BranchAndCutSolver
from logic.src.policies.branch_and_cut.policy_bc import PolicyBC
from logic.src.policies.branch_and_cut.vrpp_model import VRPPModel

__all__ = ["VRPPModel", "BranchAndCutSolver", "PolicyBC"]
