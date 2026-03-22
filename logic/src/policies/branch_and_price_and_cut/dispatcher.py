"""
Branch-and-Price-and-Cut (BPC) solver dispatcher.

Reference:
    Laporte, G., Hane, C. A., & Vance, P. H. "USING BRANCH-AND-PRICE-AND-CUT
    TO SOLVE ORIGIN-DESTINATION INTEGER MULTICOMMODITY FLOW PROBLEMS.", 1998.
"""

import ConfigSpace.api.types.categorical
from typing import Optional

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .bpc_engine import run_internal_bpc
from .gurobi_engine import run_bpc_gurobi
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
    expand_pool=False,
    profit_aware_operators=False,
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
        expand_pool (bool, optional): Whether to expand the pool of routes
        profit_aware_operators (bool, optional): Whether to use profit-aware operators
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
    elif engine == "internal":
        return run_internal_bpc(
            dist_matrix, wastes, capacity, R, C, values, must_go_indices, expand_pool, profit_aware_operators, recorder
        )
    else:
        raise ValueError(f"Unknown BPC engine: {engine}")
