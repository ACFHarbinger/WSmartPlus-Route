"""
Vehicle Routing Problem with Profits (VRPP) policy module.

This module implements a high-level wrapper for VRPP-based waste collection.
It handles the prediction of which bins will overflow (must-go bins) and
dispatches to the appropriate solver (Gurobi or Hexaly).

VRPP formulation:
- Objective: Maximize (Revenue - Travel Cost - Vehicle Cost)
- Nodes have profits (collected waste revenue)
- Nodes can be optionally visited (prize collecting)
- Must-go nodes: Bins predicted to overflow (high penalty if skipped)
- Capacity constraints on all vehicles

The policy uses statistical prediction (mean + σ * std) to identify
critical bins that require collection.
"""

from typing import Any, List, Optional, Tuple

import gurobipy as gp
import numpy as np
from numpy.typing import NDArray

from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params
from logic.src.policies.adapters import IPolicy, PolicyRegistry
from logic.src.policies.single_vehicle import (
    find_route,
    get_route_cost,
    local_search_2opt,
)
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
    config: Optional[dict] = None,
) -> Tuple[Optional[List[int]], float, float]:
    """
    Execute VRPP-based waste collection policy.

    Predicts which bins will overflow and solves a Prize-Collecting VRP to
    maximize profit (revenue from collected waste minus travel and vehicle costs).

    Must-go prediction: bins where (current + mean + param*std) >= 100%

    Args:
        policy (str): Policy name (e.g., 'gurobi_vrpp_0.5', 'hexaly_vrpp_1.0').
            Format: '{optimizer}_vrpp_{param}' where param is the std multiplier
        bins_c (NDArray[np.float64]): Current bin fill levels (0-100%)
        bins_means (NDArray[np.float64]): Mean daily accumulation rates
        bins_std (NDArray[np.float64]): Std deviation of accumulation rates
        distance_matrix (List[List[float]]): Distance matrix (N x N)
        model_env (Optional[gp.Env]): Gurobi environment (for Gurobi only)
        waste_type (str): Waste type identifier (e.g., 'plastic')
        area (str): Geographic area identifier (e.g., 'riomaior')
        n_vehicles (int): Number of vehicles available
        config (Optional[dict]): Additional configuration parameters

    Returns:
        Tuple[Optional[List[int]], float, float]: Routes, profit, and cost
            - routes: Sequence of node IDs including depot returns, or None if failed
            - profit: Total profit (revenue - cost)
            - cost: Total travel cost

    Raises:
        ValueError: If policy format is invalid or param <= 0
    """
    optimizer = "gurobi" if "gurobi" in policy else "hexaly"
    try:
        param_str = policy.rsplit("_vrpp", 1)[1]
        if param_str.startswith("_"):
            param_str = param_str[1:]
        param = float(param_str)
    except (IndexError, ValueError):
        raise ValueError(f"Invalid policy format: {policy}. Expected format like 'gurobi_vrpp0.1' or 'gurobi_vrpp_0.1'")

    if param <= 0:
        raise ValueError(f"Invalid param value for {optimizer}_vrpp: {param}")

    # Load parameters
    Q, R, B, C, V = load_area_and_waste_type_params(area, waste_type)
    values = {
        "Q": Q,
        "R": R,
        "B": B,
        "C": C,
        "V": V,
        "Omega": 0.1,
        "delta": 0,
        "psi": 1,
    }

    if config:
        values.update(config)

    n_bins = len(bins_c)
    binsids = np.arange(0, n_bins + 1).tolist()

    # Calculate must_go bins
    must_go = []
    for container_id in range(n_bins):
        pred_value = bins_c[container_id] + bins_means[container_id] + param * bins_std[container_id]
        if pred_value >= 100:
            must_go.append(container_id + 1)  # +1 because of depot at 0

    time_limits = [600, 3600] if optimizer == "gurobi" else [60, 360]

    routes = None
    profit = 0.0
    cost = 0.0
    for tl in time_limits:
        try:
            # For Hexaly, we don't pass the Gurobi env
            current_env = model_env if optimizer == "gurobi" else None

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
                number_vehicles=n_vehicles,
            )

            if res_routes and res_cost > 0:
                routes = res_routes
                profit = res_profit
                cost = res_cost
                break
        except Exception:
            # Continue to allow retry with higher time limit
            continue

    return routes, profit, cost


@PolicyRegistry.register("policy_vrpp")
class VRPPPolicy(IPolicy):
    """
    VRPP (Vehicle Routing Problem with Profits) policy class.
    Executes Prize-Collecting VRP using Gurobi or Hexaly solvers.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the VRPP policy.
        """
        policy = kwargs["policy"]
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        model_env = kwargs["model_env"]
        waste_type = kwargs["waste_type"]
        area = kwargs["area"]
        n_vehicles = kwargs["n_vehicles"]
        distancesC = kwargs["distancesC"]
        run_tsp = kwargs["run_tsp"]
        two_opt_max_iter = kwargs.get("two_opt_max_iter", 0)
        config = kwargs.get("config", {})

        vrpp_config = config.get("vrpp", {})

        routes, _, _ = policy_vrpp(
            policy,
            bins.c,
            bins.means,
            bins.std,
            distance_matrix.tolist(),
            model_env,
            waste_type,
            area,
            n_vehicles,
            config=vrpp_config,
        )
        tour = []
        cost = 0
        if routes:
            tour = find_route(distancesC, np.array(routes)) if run_tsp else routes
            if two_opt_max_iter > 0:
                tour = local_search_2opt(tour, distance_matrix, two_opt_max_iter)
            cost = get_route_cost(distance_matrix, tour)
        return tour, cost, None
