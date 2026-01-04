import numpy as np
import gurobipy as gp

from numpy.typing import NDArray
from typing import List, Optional, Tuple
from logic.src.pipeline.simulator.loader import load_area_and_waste_type_params
from logic.src.policies.vrpp_optimizer import run_vrpp_optimizer


def policy_vrpp(
    policy: str,
    bins_c: NDArray[np.float64],
    bins_means: NDArray[np.float64],
    bins_std: NDArray[np.float64],
    distance_matrix: List[List[float]],
    model_env: Optional[gp.Env],
    waste_type: str,
    area: str,
    n_vehicles: int,
    config: Optional[dict] = None
) -> Tuple[Optional[List[int]], float, float]:
    """
    High-level policy wrapper for VRPP.
    Calculates must_go bins and dispatches to the appropriate optimizer (Gurobi/Hexaly).
    """
    optimizer = 'gurobi' if 'gurobi' in policy else 'hexaly'
    try:
        param_str = policy.rsplit("_vrpp", 1)[1]
        if param_str.startswith('_'):
            param_str = param_str[1:]
        param = float(param_str)
    except (IndexError, ValueError):
        raise ValueError(f"Invalid policy format: {policy}. Expected format like 'gurobi_vrpp0.1' or 'gurobi_vrpp_0.1'")
        
    if param <= 0:
        raise ValueError(f'Invalid param value for {optimizer}_vrpp: {param}')

    # Load parameters
    Q, R, B, C, V = load_area_and_waste_type_params(area, waste_type)
    values = {
        'Q': Q, 'R': R, 'B': B, 'C': C, 'V': V, 
        'Omega': 0.1, 'delta': 0, 'psi': 1
    }
    
    if config:
        values.update(config)

    n_bins = len(bins_c)
    binsids = np.arange(0, n_bins + 1).tolist()
    
    # Calculate must_go bins
    must_go = []
    for container_id in range(n_bins):
        pred_value = (bins_c[container_id] + bins_means[container_id] + param * bins_std[container_id])
        if pred_value >= 100:
            must_go.append(container_id + 1) # +1 because of depot at 0

    time_limits = [600, 3600] if optimizer == 'gurobi' else [60, 360]
    
    routes = None
    profit = 0.0
    cost = 0.0
    for tl in time_limits:
        try:
            # For Hexaly, we don't pass the Gurobi env
            current_env = model_env if optimizer == 'gurobi' else None
            
            res_routes, res_profit, res_cost = run_vrpp_optimizer(
                bins=bins_c,
                distance_matrix=distance_matrix,
                param=param,
                media=bins_means,
                desviopadrao=bins_std,
                values=values,
                binsids=binsids,
                must_go=must_go,
                env=current_env,
                optimizer=optimizer,
                time_limit=tl,
                number_vehicles=n_vehicles
            )
            
            if res_routes and res_cost > 0:
                routes = res_routes
                profit = res_profit
                cost = res_cost
                break
        except Exception as e:
            # Continue to allow retry with higher time limit
            continue
            
    return routes, profit, cost
