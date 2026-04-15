"""
Consolidated Exact Solvers for VRPP.

Provides high-performance, mathematically rigorous implementations of
Branch-and-Price-and-Cut components.
"""

from logic.src.policies.helpers.branching_solvers.branching import (
    AnyBranchingConstraint,
    BranchAndBoundTree,
    EdgeBranching,
    MultiEdgePartitionBranching,
    RyanFosterBranching,
)
from logic.src.policies.helpers.branching_solvers.common import BranchNode, Route
from logic.src.policies.helpers.branching_solvers.master_problem import GlobalCutPool, VRPPMasterProblem
from logic.src.policies.helpers.branching_solvers.pricing import Label, RCSPPSolver
from logic.src.policies.helpers.branching_solvers.separation import (
    CapacityCut,
    CombInequality,
    PCSubtourEliminationCut,
    SeparationEngine,
)

from .lagrangian_relaxation.subgradient_optimization import _nearest_neighbour_tour_cost, run_subgradient
from .lagrangian_relaxation.uncapacitated_orienteering_problem import solve_uncapacitated_op

__all__ = [
    "AnyBranchingConstraint",
    "BranchAndBoundTree",
    "BranchNode",
    "EdgeBranching",
    "RyanFosterBranching",
    "MultiEdgePartitionBranching",
    "GlobalCutPool",
    "Route",
    "VRPPMasterProblem",
    "RCSPPSolver",
    "Label",
    "CapacityCut",
    "CombInequality",
    "PCSubtourEliminationCut",
    "SeparationEngine",
    "run_subgradient",
    "_nearest_neighbour_tour_cost",
    "solve_uncapacitated_op",
]
