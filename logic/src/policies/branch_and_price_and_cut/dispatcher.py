"""
Branch-and-Price-and-Cut (BPC) solver dispatcher.

Reference:
    Barnhart, C., Johnson, E. L., Nemhauser, G. L., Savelsbergh, M. W., & Vance, P. H. (1998).
    "Branch-and-price: Column generation for solving huge integer programs."
    Operations Research, 46(3), 316-329.

Note:
    This dispatcher coordinates multiple solver backends for VRPP. The 'custom'
    engine implements an exact BPC algorithm adapted for the VRPP context,
    following high-level sequencing (converged CG, cut separation, branching)
    found in the BPC framework of Barnhart et al. (1998).
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
    node_coords=None,
    recorder: Optional[PolicyStateRecorder] = None,
):
    """
    Main dispatcher for Branch-and-Price-and-Cut solvers.

    Selects and runs the appropriate BPC solver based on configuration.
    The 'custom' engine provides an exact BPC implementation specifically
    adapted for Waste-Collecting CVRP and VRPP from the Barnhart et al. framework.

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
    # Fix 7: Warn if ignored parameters are passed to non-custom engines
    if engine in ("vrpy", "gurobi", "ortools") and (expand_pool or profit_aware_operators):
        import warnings

        warnings.warn(
            f"BPC engine '{engine}' does not support expand_pool or "
            f"profit_aware_operators. These parameters are ignored. "
            "Use engine='custom' to enable them.",
            stacklevel=2,
        )

    if engine == "vrpy":
        return run_bpc_vrpy(dist_matrix, wastes, capacity, R, C, values, recorder=recorder)
    elif engine == "gurobi":
        return run_bpc_gurobi(dist_matrix, wastes, capacity, R, C, values, must_go_indices, env, recorder=recorder)
    elif engine == "ortools":
        return run_bpc_ortools(dist_matrix, wastes, capacity, R, C, values, must_go_indices, recorder=recorder)
    elif engine == "custom":
        return run_custom_bpc(
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            values,
            must_go_indices,
            node_coords,
            expand_pool,
            profit_aware_operators,
            recorder,
        )
    else:
        raise ValueError(f"Unknown BPC engine: {engine}")
