"""
Policy adapter for Branch-and-Cut VRPP solver.

Integrates the Branch-and-Cut algorithm into the WSmart+ Route policy framework.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from logic.src.policies.branch_and_cut.solver import GUROBI_AVAILABLE, BranchAndCutSolver
from logic.src.policies.branch_and_cut.vrpp_model import VRPPModel


class PolicyBC:
    """
    Branch-and-Cut policy for VRPP.

    This policy uses an exact branch-and-cut algorithm to solve the Vehicle Routing
    Problem with Profits optimally (within time limits and MIP gap tolerance).
    """

    def __init__(
        self,
        time_limit: float = 60.0,
        mip_gap: float = 0.01,
        max_cuts_per_round: int = 50,
        use_heuristics: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize Branch-and-Cut policy.

        Args:
            time_limit: Maximum solving time in seconds.
            mip_gap: Relative MIP gap tolerance.
            max_cuts_per_round: Maximum cuts to add per separation round.
            use_heuristics: Whether to use primal heuristics for warm start.
            verbose: Print detailed solving information.
        """
        if not GUROBI_AVAILABLE:
            raise ImportError(
                "Gurobi is required for Branch-and-Cut solver. Please ensure Gurobi is installed and licensed."
            )

        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.max_cuts_per_round = max_cuts_per_round
        self.use_heuristics = use_heuristics
        self.verbose = verbose

        self.name = "branch_and_cut"
        self.stats: Dict[str, Any] = {}

    def __call__(
        self,
        coords: pd.DataFrame,
        must_go: List[int],
        distance_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        **kwargs,
    ) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        Solve VRPP instance using Branch-and-Cut.

        Args:
            coords: DataFrame with node coordinates (unused, we use distance_matrix).
            must_go: List of mandatory node indices to visit.
            distance_matrix: N×N distance matrix.
            wastes: Dictionary mapping node index to waste/demand.
            capacity: Vehicle capacity.
            R: Revenue per kg of waste collected.
            C: Cost per km traveled.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (tour, profit, statistics).
        """
        n_nodes = len(distance_matrix)

        # Build VRPP model
        model = VRPPModel(
            n_nodes=n_nodes,
            cost_matrix=distance_matrix,
            wastes=wastes,
            capacity=capacity,
            revenue_per_kg=R,
            cost_per_km=C,
            mandatory_nodes=set(must_go) if must_go else set(),
        )

        # Create solver
        solver = BranchAndCutSolver(
            model=model,
            time_limit=self.time_limit,
            mip_gap=self.mip_gap,
            max_cuts_per_round=self.max_cuts_per_round,
            use_heuristics=self.use_heuristics,
            verbose=self.verbose,
        )

        # Solve
        tour, profit, stats = solver.solve()

        # Validate solution
        is_valid, message = model.validate_tour(tour)
        if not is_valid and self.verbose:
            print(f"Warning: Invalid solution - {message}")

        # Store statistics
        self.stats = stats
        self.stats["valid"] = is_valid
        self.stats["validation_message"] = message

        # Convert profit to cost for consistency with other policies
        # (other policies minimize cost, so we return negative profit)
        cost = -profit if profit > 0 else 0.0

        return tour, cost, self.stats


def run_branch_and_cut(
    coords: pd.DataFrame,
    must_go: List[int],
    distance_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    time_limit: float = 60.0,
    mip_gap: float = 0.01,
    verbose: bool = False,
    **kwargs,
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    Convenience function for running Branch-and-Cut solver.

    Args:
        coords: DataFrame with node coordinates.
        must_go: List of mandatory node indices.
        distance_matrix: N×N distance matrix.
        wastes: Dictionary mapping node index to waste/demand.
        capacity: Vehicle capacity.
        R: Revenue per kg.
        C: Cost per km.
        time_limit: Maximum solving time.
        mip_gap: MIP gap tolerance.
        verbose: Print detailed output.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (tour, cost, statistics).
    """
    policy = PolicyBC(
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
    )

    return policy(
        coords=coords,
        must_go=must_go,
        distance_matrix=distance_matrix,
        wastes=wastes,
        capacity=capacity,
        R=R,
        C=C,
        **kwargs,
    )
