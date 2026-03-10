"""
HGS Dispatcher module.

Dispatches to different HGS implementations based on configuration.
"""

from .hgs import HGSSolver
from .params import HGSParams
from .pyvrp_wrapper import solve_pyvrp


def run_hgs(dist_matrix, wastes, capacity, R, C, values, mandatory_nodes=None, *args):
    """
    Main HGS entry point with dispatching logic.

    Args:
        dist_matrix: Distance matrix.
        wastes: Bin wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Dictionary of parameters and config.
        mandatory_nodes: List of local node indices that MUST be visited.
        *args: Additional arguments (ignored or passed through).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    engine = values.get("engine") or values.get("variant")
    if engine == "pyvrp":
        return solve_pyvrp(dist_matrix, wastes, capacity, R, C, values)

    if len(dist_matrix) <= 1:
        return [], 0.0, 0.0

    if len(dist_matrix) == 2:
        d = wastes.get(1, 0)
        if d <= capacity:
            # Calculate simple profit/cost
            cost = dist_matrix[0][1] + dist_matrix[1][0]
            profit = d * R
            return [[1]], profit, C * cost
        else:
            return [], 0.0, 0.0

    params = HGSParams(
        time_limit=values.get("time_limit", 10),
        population_size=values.get("population_size", 50),
        elite_size=values.get("elite_size", 10),
        mutation_rate=values.get("mutation_rate", 0.2),
        n_generations=values.get("n_generations", 100),
        alpha_diversity=values.get("alpha_diversity", 0.5),
        min_diversity=values.get("min_diversity", 0.2),
        diversity_change_rate=values.get("diversity_change_rate", 0.05),
        no_improvement_threshold=values.get("no_improvement_threshold", 20),
        survivor_threshold=values.get("survivor_threshold", 2),
        max_vehicles=values.get("max_vehicles", 0),
        local_search_iterations=values.get("local_search_iterations", 500),
        crossover_rate=values.get("crossover_rate", 0.7),
    )
    solver = HGSSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes, seed=values.get("seed"))
    return solver.solve()
