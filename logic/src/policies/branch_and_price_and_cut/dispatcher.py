"""
Branch-and-Price-and-Cut (BPC) solver dispatcher.

Reference:
    Barnhart, C., Hane, C. A., & Vance, P. H. (1998).
    "Using Branch-and-Price-and-Cut to Solve Origin-Destination Integer
    Multicommodity Flow Problems." Operations Research, 48(2), 318-326.

Note:
    This dispatcher coordinates multiple solver backends:
    - 'bpc_native': True Branch-and-Price-and-Cut (Column Generation + B&B tree)
    - 'gurobi': Standard Branch-and-Cut (NOT BPC - uses Gurobi's built-in B&C)
    - 'ortools': Constraint Programming / Local Search (NOT BPC)
    - 'vrpy': VRPy library wrapper (Column Generation without full BPC)
"""

from typing import Optional

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .bpc_engine import run_custom_bpc
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
    elif engine == "ortools":
        return run_bpc_ortools(dist_matrix, wastes, capacity, R, C, values, must_go_indices, recorder=recorder)
    elif engine == "custom":
        return run_custom_bpc(
            dist_matrix, wastes, capacity, R, C, values, must_go_indices, expand_pool, profit_aware_operators, recorder
        )
    else:
        raise ValueError(f"Unknown BPC engine: {engine}")
