"""
Branch-and-Price-and-Cut (BPC) solver dispatcher.

Reference:
    Laporte, G., Hane, C. A., & Vance, P. H. "USING BRANCH-AND-PRICE-AND-CUT
    TO SOLVE ORIGIN-DESTINATION INTEGER MULTICOMMODITY FLOW PROBLEMS.", 1998.
"""

from typing import Optional

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .gurobi_engine import run_bpc_gurobi
from .ortools_engine import run_bpc_ortools
from .vrpy_engine import run_bpc_vrpy


def run_bpc(
    dist_matrix,
    wastes,
    capacity,
    R,
    C,
    values,
    must_go_indices=None,
    env=None,
    recorder: Optional[PolicyStateRecorder] = None,
):
    """
    Main dispatcher for Branch-and-Price-and-Cut solvers.

    Selects and runs the appropriate BPC solver based on configuration.
    Supports Waste-Collecting CVRP with optional must-go nodes.

    Args:
        dist_matrix (np.ndarray): Distance matrix (N x N) with depot at index 0
        wastes (dict): Node wastes {node_id: waste_value}
        capacity (float): Vehicle capacity constraint
        R (float): Revenue per unit waste
        C (float): Cost per unit distance
        values (dict): Configuration with 'bpc_engine' in ['ortools', 'vrpy', 'gurobi'].
            Default: 'ortools'. Also supports 'time_limit' (default: 30 seconds)
        must_go_indices (set, optional): Node IDs that must be visited
        env (gp.Env, optional): Gurobi environment (for Gurobi engine only)
        recorder (PolicyStateRecorder, optional): Telemetry recorder.

    Returns:
        Tuple[List[List[int]], float]: Routes and total cost
            - routes: List of routes, each containing node IDs
            - cost: Total travel cost (distance * C)
    """
    engine = values.get("bpc_engine", "ortools")

    if engine == "vrpy":
        return run_bpc_vrpy(dist_matrix, wastes, capacity, R, C, values, recorder=recorder)
    elif engine == "gurobi":
        return run_bpc_gurobi(dist_matrix, wastes, capacity, R, C, values, must_go_indices, env, recorder=recorder)
    else:
        # Default to OR-Tools
        return run_bpc_ortools(dist_matrix, wastes, capacity, R, C, values, must_go_indices, recorder=recorder)
