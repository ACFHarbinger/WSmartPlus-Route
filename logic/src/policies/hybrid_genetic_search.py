"""
Hybrid Genetic Search (HGS) policy module.

Combines genetic algorithms with local search and the Split algorithm
for solving the Capacitated Vehicle Routing Problem with Profits.

Reference:
    Vidal, T., Crainic, T. G., Gendreau, M., & Prins, C. (2016). A unified
    solution framework for multi-attribute vehicle routing problems. European
    Journal of Operational Research, 234(3), 658-673.
"""

import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.policies.adapters import IPolicy, PolicyRegistry

from .hgs_aux.evolution import evaluate, ordered_crossover, update_biased_fitness
from .hgs_aux.local_search import LocalSearch
from .hgs_aux.pyvrp_wrapper import solve_pyvrp
from .hgs_aux.split import LinearSplit
from .hgs_aux.types import HGSParams, Individual


class HGSSolver:
    """
    Implements Hybrid Genetic Search for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        demands: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HGSParams,
    ):
        """
        Initialize the HGS solver.

        Args:
            dist_matrix: NxN distance matrix.
            demands: Dictionary of node demands.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Detailed HGS parameters.
        """
        self.d = dist_matrix
        self.demands = demands
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        self.split_manager = LinearSplit(dist_matrix, demands, capacity, R, C, params.max_vehicles)
        self.ls = LocalSearch(dist_matrix, demands, capacity, R, C, params)

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Hybrid Genetic Search algorithm.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        population: List[Individual] = []

        # 1. Initial Population
        for _ in range(self.params.population_size):
            gt = self.nodes[:]
            random.shuffle(gt)
            ind = Individual(gt)
            evaluate(ind, self.split_manager)
            population.append(ind)

        update_biased_fitness(population, self.params.elite_size)

        start_time = time.time()
        it = 0
        while time.time() - start_time < self.params.time_limit:
            it += 1
            # 2. Selection & Crossover
            p1, p2 = self._select_parents(population)
            child = ordered_crossover(p1, p2)

            # 3. Local Search (Mutation)
            if random.random() < self.params.mutation_rate:
                evaluate(child, self.split_manager)
                child = self.ls.optimize(child)

            evaluate(child, self.split_manager)
            population.append(child)

            # 4. Survivor Selection
            if len(population) > self.params.population_size * 2:
                update_biased_fitness(population, self.params.elite_size)
                population.sort(key=lambda x: x.fitness)
                population = population[: self.params.population_size]

        update_biased_fitness(population, self.params.elite_size)
        best_ind = min(population, key=lambda x: -x.profit_score)

        return best_ind.routes, best_ind.profit_score, best_ind.cost

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        # Binary Tournament
        def tournament():
            """Perform a binary tournament selection."""
            i1, i2 = random.sample(population, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()


def run_hgs(dist_matrix, demands, capacity, R, C, values, *args):
    """
    Main HGS entry point with dispatching logic.

    Args:
        dist_matrix: Distance matrix.
        demands: Bin demands.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Dictionary of parameters and config.
        *args: Additional arguments (ignored or passed through).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    engine = values.get("engine") or values.get("variant")
    if engine == "pyvrp":
        return solve_pyvrp(dist_matrix, demands, capacity, R, C, values)

    params = HGSParams(
        time_limit=values.get("time_limit", 10),
        population_size=values.get("population_size", 50),
        max_vehicles=values.get("max_vehicles", 0),
    )
    solver = HGSSolver(dist_matrix, demands, capacity, R, C, params)
    return solver.solve()


@PolicyRegistry.register("policy_hgs")
class HGSPolicy(IPolicy):
    """
    Hybrid Genetic Search policy class.
    Executes HGS for VRP.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the HGS policy.
        """
        policy = kwargs["policy"]
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        kwargs["waste_type"]
        kwargs["area"]
        config = kwargs.get("config", {})

        # 1. Determine Must-Go Bins (VRPP Logic)
        try:
            # Pattern: policy_hgs_<threshold>
            threshold_std = float(policy.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            threshold_std = 1.0  # Default

        if not hasattr(bins, "means") or bins.means is None:
            raise ValueError("Bins object missing 'means' attribute.")
        else:
            means = bins.means
            std = bins.std

        current_fill = bins.c
        predicted_fill = current_fill + means + (threshold_std * std)

        # Must-go bins
        must_go_indices = np.where((predicted_fill >= 100.0) | (current_fill >= 100.0))[0].tolist()

        # 2. Prepare Data for HGS
        # HGS visits all nodes in the passed matrix/demands.
        # We must subset for just the target bins.

        target_nodes = must_go_indices
        if not target_nodes:
            return [0, 0], 0.0, None

        # Build demands dict {node_idx: demand}
        # Note: demands dict keys will be 1..K (new indices)
        demands = {i + 1: current_fill[i] for i in range(len(current_fill))}  # Full demands

        # HGS Config
        hgs_config = config.get("hgs", {})
        capacity = hgs_config.get("capacity", 100.0)
        revenue = hgs_config.get("revenue", 1.0)
        cost_unit = hgs_config.get("cost_unit", 1.0)

        # Subset mapping
        real_target_indices = [idx + 1 for idx in target_nodes]
        subset_indices = [0] + real_target_indices

        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]

        sub_demands = {i: demands[original_idx] for i, original_idx in enumerate(real_target_indices, 1)}

        # Run HGS
        best_routes, _, _ = run_hgs(sub_dist_matrix, sub_demands, capacity, revenue, cost_unit, hgs_config)

        # Map routes back
        tour = [0]
        if best_routes:
            for route in best_routes:
                for node_idx in route:
                    original_matrix_idx = subset_indices[node_idx]
                    tour.append(original_matrix_idx)
                tour.append(0)

        if len(tour) == 1:
            tour = [0, 0]

        # Recalculate cost
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += distance_matrix[tour[i]][tour[i + 1]]

        return tour, cost, None
