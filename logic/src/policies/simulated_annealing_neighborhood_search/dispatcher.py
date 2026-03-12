"""
Simulated Annealing Neighborhood Search (SANS) Dispatcher Module.

This module contains the execution logic for Simulated Annealing Neighborhood Search (SANS).
It provides functions to run both the improved ('new') and original ('og') SA engines.

Reference:
    Jorge, D., Antunes, A. P., Ramos, T. R. P., & Barbosa-Povoa, A. P.
    "A hybrid metaheuristic for smart waste collection problems with
    workload concerns", 2022.
"""

import random
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from logic.src.data.processor import convert_to_dict
from logic.src.policies.simulated_annealing_neighborhood_search import (
    improved_simulated_annealing,
)
from logic.src.policies.simulated_annealing_neighborhood_search.common.routes import create_points
from logic.src.policies.simulated_annealing_neighborhood_search.common.solution_initialization import (
    compute_initial_solution,
)
from logic.src.policies.simulated_annealing_neighborhood_search.refinement.route_search import find_solutions
from logic.src.policies.travelling_salesman_problem.tsp import get_route_cost


def execute_new(policy: Any, **kwargs: Any) -> Tuple[List[int], float, Any]:
    """Execute the improved simulated annealing engine.

    Args:
        policy: The SANSPolicy instance.
        **kwargs: Arguments passed to the policy's execute method.

    Returns:
        Tuple[List[int], float, Any]: Tour, distance, and metadata.
    """
    must_go = kwargs.get("must_go", [])
    early_result = policy._validate_must_go(must_go)
    if early_result is not None:
        return early_result

    bins = kwargs["bins"]
    distance_matrix = kwargs["distance_matrix"]
    coords = kwargs["coords"]
    area = kwargs.get("area", "Rio Maior")
    waste_type = kwargs.get("waste_type", "plastic")
    config = kwargs.get("config", {})

    # Load area parameters
    Q, R, C, values = policy._load_area_params(area, waste_type, config)
    policy._log_solver_params(values, kwargs)

    B = values.get("B", 0)
    V = values.get("V", 0)

    # Prepare data for SANS logic
    data = pd.DataFrame(
        {
            "ID": np.arange(1, bins.n + 1),
            "#bin": np.arange(1, bins.n + 1),
            "Stock": bins.c.astype("float32"),
            "Accum_Rate": bins.means.astype("float32"),
        }
    )
    # Add depot
    depot_row = pd.DataFrame([{"#bin": 0, "Stock": 0.0, "Accum_Rate": 0.0}])
    data = pd.concat([depot_row, data], ignore_index=True)

    coords_dict = convert_to_dict(coords)
    id_to_index = {i: i for i in range(len(distance_matrix))}

    initial_routes = compute_initial_solution(data, coords_dict, distance_matrix, Q, id_to_index)
    current_route = initial_routes[0] if initial_routes else [0, 0]

    # Ensure must_go bins are in the route
    must_go_1 = list(must_go)
    route_set = set(current_route)
    for b in must_go_1:
        if b not in route_set:
            current_route.insert(1, b)

    # Get SA parameters from typed config or raw config
    rng = random.Random(kwargs.get("seed")) if kwargs.get("seed") is not None else random.Random()
    sans_config = config.get("sans", {})
    T_init = sans_config.get("T_init", 75)
    iterations_per_T = sans_config.get("iterations_per_T", 5000)
    alpha = sans_config.get("alpha", 0.95)
    T_min = sans_config.get("T_min", 0.01)
    time_limit = values.get("time_limit", 60)
    perc_overflow = sans_config.get("perc_bins_can_overflow", 0.0)

    optimized_routes, best_profit, last_distance, _, _ = improved_simulated_annealing(
        [current_route],
        distance_matrix,
        time_limit,
        id_to_index,
        data,
        Q,
        T_init,
        T_min,
        alpha,
        iterations_per_T,
        R,
        V,
        B,
        C,
        must_go_1,
        perc_bins_can_overflow=perc_overflow,
        volume=V,
        density_val=B,
        max_vehicles=1,
        rng=rng,
    )

    tour = optimized_routes[0] if optimized_routes else [0, 0]
    return tour, last_distance, {"profit": best_profit}


def execute_og(policy: Any, **kwargs: Any) -> Tuple[List[int], float, Any]:
    """Execute the original LAC (look-ahead collection) engine.

    Args:
        policy: The SANSPolicy instance.
        **kwargs: Arguments passed to the policy's execute method.

    Returns:
        Tuple[List[int], float, Any]: Tour, distance, and metadata.
    """
    from logic.src.constants.routing import (
        DEFAULT_COMBINATION,
        DEFAULT_SHIFT_DURATION,
        DEFAULT_TIME_LIMIT,
        DEFAULT_V_VALUE,
    )

    must_go = kwargs.get("must_go", [])
    early_result = policy._validate_must_go(must_go)
    if early_result is not None:
        return early_result

    bins = kwargs["bins"]
    distance_matrix = kwargs["distance_matrix"]
    coords = kwargs["coords"]
    new_data = kwargs.get("new_data")  # Expecting the dataframe
    area = kwargs.get("area", "Rio Maior")
    waste_type = kwargs.get("waste_type", "plastic")
    config = kwargs.get("config", {})
    lac_config = config.get("lac", config.get("sans", {}))

    # Load area parameters
    Q, R, C, area_values = policy._load_area_params(area, waste_type, config)
    policy._log_solver_params(area_values, kwargs)
    V = lac_config.get("V", DEFAULT_V_VALUE)

    values = {
        "Q": Q,
        "R": R,
        "B": area_values.get("B", 0),
        "C": C,
        "V": V,
        "shift_duration": DEFAULT_SHIFT_DURATION,
        "perc_bins_can_overflow": 0,
    }
    values.update(lac_config)

    # must_go bins are 0-based, find_solutions expects 1-based
    must_go_1 = [b + 1 for b in must_go]

    # Handle case where new_data might not be passed
    if new_data is None:
        # Build minimal dataframe similar to _execute_new
        new_data = pd.DataFrame(
            {
                "ID": np.arange(1, bins.n + 1),
                "#bin": np.arange(1, bins.n + 1),
                "Stock": bins.c.astype("float32"),
                "Accum_Rate": bins.means.astype("float32"),
            }
        )
        depot_row = pd.DataFrame([{"#bin": 0, "Stock": 0.0, "Accum_Rate": 0.0}])
        new_data = pd.concat([depot_row, new_data], ignore_index=True)

    points = create_points(new_data, coords)

    # Get combination and time_limit from typed config or raw config
    cfg = policy._config
    combination = (
        cfg.combination if cfg is not None and cfg.combination else lac_config.get("combination", DEFAULT_COMBINATION)
    )
    og_time_limit = cfg.time_limit if cfg is not None else lac_config.get("time_limit", DEFAULT_TIME_LIMIT)

    rng = random.Random(kwargs.get("seed")) if kwargs.get("seed") is not None else random.Random()
    np_rng = np.random.RandomState(kwargs.get("seed")) if kwargs.get("seed") is not None else np.random.RandomState()
    try:
        res, _, _ = find_solutions(
            new_data,
            coords,
            distance_matrix,
            combination,
            must_go_1,
            values,
            bins.n,
            points,
            time_limit=og_time_limit,
            rng=rng,
            np_rng=np_rng,
        )
    except Exception:
        return [0, 0], 0.0, None

    tour = res[0] if res else [0, 0]
    cost = get_route_cost(distance_matrix, tour)

    # Compute profit: collected revenue - distance cost
    visited = {n for n in tour if n != 0}
    collected_revenue = sum(float(bins.c[n - 1]) * R for n in visited if 1 <= n <= bins.n)
    profit = collected_revenue - cost * C

    return tour, cost, {"profit": profit}
