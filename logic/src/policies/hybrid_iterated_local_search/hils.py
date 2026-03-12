"""
Hybrid Iterated Local Search (HILS) solver.

Combines the Iterated Local Search (ILS) metaheuristic with Randomized
Variable Neighborhood Descent (RVND) and an exact Set Partitioning (SP) formulation.

Reference:
    Ropke, S., & Pisinger, D. "A unified heuristic for a large
    class of Vehicle Routing Problems with Backhauls", 2006
"""

import copy
import logging
import time
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from logic.src.policies.hybrid_iterated_local_search.params import HILSParams
from logic.src.policies.hybrid_iterated_local_search.rvnd import RVND
from logic.src.policies.other.local_search.local_search_aco import (
    ACOLocalSearch,
)
from logic.src.tracking.viz_mixin import PolicyVizMixin

logger = logging.getLogger(__name__)


class HILSSolver(PolicyVizMixin):
    """
    HILS solver executing ILS with RVND and resolving via Set Partitioning.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HILSParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the HILS solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes (demands/profits).
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Parameters for the algorithm.
            mandatory_nodes: List of mandatory nodes.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.rng = np.random.default_rng(params.seed)

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Keep track of unique valid routes explored
        # A route is mapped to a tuple for correct hashing
        self.route_pool: Set[Tuple[int, ...]] = set()

    def calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total_dist = 0.0
        for route in routes:
            if not route:
                continue
            dist = self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                dist += self.dist_matrix[route[i]][route[i + 1]]
            dist += self.dist_matrix[route[-1]][0]
            total_dist += dist
        return total_dist * self.C

    def calculate_profit(self, routes: List[List[int]]) -> float:
        """Calculate network profit (revenue - cost)."""
        cost = self.calculate_cost(routes)
        revenue = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return revenue - cost

    def _add_to_pool(self, routes: List[List[int]]):
        """Standardize and hash routes to add to the unique pool."""
        for route in routes:
            if not route:
                continue
            # Keep original orientation if that maps to a specific sequential distance cost
            # But technically for symmetric graphs we could sort boundaries
            self.route_pool.add(tuple(route))

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Perturbation mechanism for generating neighborhood jumps.
        Randomly removes perturbation_size nodes and greedily reinserts them.
        """
        all_nodes = [n for r in routes for n in r]
        if not all_nodes:
            return routes

        num_remove = min(len(all_nodes), self.params.perturbation_size)
        removed_nodes = list(self.rng.choice(all_nodes, size=num_remove, replace=False))

        new_routes = []
        for r in routes:
            cleaned = [n for n in r if n not in removed_nodes]
            if cleaned:
                new_routes.append(cleaned)

        # Greedily Reinsert
        for node in removed_nodes:
            best_route_idx = -1
            best_pos = -1
            best_cost_delta = float("inf")
            node_waste = self.wastes.get(node, 0.0)

            for i, r in enumerate(new_routes):
                current_load = sum(self.wastes.get(n, 0.0) for n in r)
                if current_load + node_waste <= self.capacity:
                    # Evaluate best insertion point
                    for j in range(len(r) + 1):
                        prev_node = r[j - 1] if j > 0 else 0
                        next_node = r[j] if j < len(r) else 0

                        delta = (
                            self.dist_matrix[prev_node][node]
                            + self.dist_matrix[node][next_node]
                            - self.dist_matrix[prev_node][next_node]
                        )
                        if delta < best_cost_delta:
                            best_cost_delta = delta
                            best_route_idx = i
                            best_pos = j

            if best_route_idx != -1:
                new_routes[best_route_idx].insert(best_pos, node)
            else:
                new_routes.append([node])

        return new_routes

    def build_initial_solution(self) -> List[List[int]]:
        """Greedy constructive heuristic."""
        nodes = self.nodes[:]
        self.rng.shuffle(nodes)
        routes = []
        curr_route = []
        load = 0.0
        mandatory_set = set(self.mandatory_nodes)

        for node in nodes:
            waste = self.wastes.get(node, 0.0)
            revenue = waste * self.R
            is_mandatory = node in mandatory_set

            if not is_mandatory and revenue < (self.dist_matrix[0][node] + self.dist_matrix[node][0]) * self.C:
                continue

            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    routes.append(curr_route)
                curr_route = [node]
                load = waste
        if curr_route:
            routes.append(curr_route)
        return routes

    def solve_set_partitioning(self) -> Tuple[List[List[int]], float, float]:
        """
        Solve the Set Partitioning Problem exactly using Gurobi over the route pool.
        Finds the optimal combination of valid routes that visits mandatory nodes
        exactly once, and optional nodes at most once, while maximizing global profit.
        """
        pool = list(self.route_pool)
        n_routes = len(pool)

        if n_routes == 0:
            return [], 0.0, 0.0

        logger.info(f"HILS solving Set Partitioning over {n_routes} unique routes.")

        # Precompute costs and node incidence
        route_costs = np.zeros(n_routes)
        route_profits = np.zeros(n_routes)

        # Map node to the indices of routes that contain it
        node_to_routes = {n: [] for n in self.nodes}

        for i, route in enumerate(pool):
            cost = self.dist_matrix[0][route[0]]
            for j in range(len(route) - 1):
                cost += self.dist_matrix[route[j]][route[j + 1]]
            cost += self.dist_matrix[route[-1]][0]
            route_costs[i] = cost * self.C

            revenue = sum(self.wastes.get(n, 0.0) * self.R for n in route)
            route_profits[i] = revenue - route_costs[i]

            for node in route:
                node_to_routes[node].append(i)

        # Build Model
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("HILS_SetPartitioning", env=env)

        model.setParam("TimeLimit", self.params.sp_time_limit)
        model.setParam("MIPGap", self.params.sp_mip_gap)

        # Variables: x[i] = 1 if route i is selected
        x = model.addVars(n_routes, vtype=GRB.BINARY, name="x")

        # Maximize Profit (Revenue - Cost)
        model.setObjective(gp.quicksum(route_profits[i] * x[i] for i in range(n_routes)), GRB.MAXIMIZE)

        # Constraints: Mandatory nodes visited exactly once
        mandatory_set = set(self.mandatory_nodes)
        for node in mandatory_set:
            if node_to_routes[node]:
                model.addConstr(gp.quicksum(x[i] for i in node_to_routes[node]) == 1, name=f"Mandatory_{node}")

        # Constraints: Optional nodes visited at most once
        optional_set = set(self.nodes) - mandatory_set
        for node in optional_set:
            if node_to_routes[node]:
                model.addConstr(gp.quicksum(x[i] for i in node_to_routes[node]) <= 1, name=f"Optional_{node}")

        model.optimize()

        if model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT} and model.SolCount > 0:
            best_routes = []
            for i in range(n_routes):
                if x[i].X > 0.5:
                    best_routes.append(list(pool[i]))

            best_cost = self.calculate_cost(best_routes)
            best_profit = self.calculate_profit(best_routes)
            return best_routes, best_profit, best_cost

        logger.warning("SP failed to find a valid combination. Falling back to best ILS solution.")
        return [], -float("inf"), float("inf")

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """Main HILS execution loop."""

        # 1. Initialize
        current_routes = copy.deepcopy(initial_solution) if initial_solution else self.build_initial_solution()
        best_routes = copy.deepcopy(current_routes)

        current_profit = self.calculate_profit(current_routes)
        best_profit = current_profit

        ls_manager = ACOLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=self.params,
            seed=self.params.seed,
        )
        rvnd = RVND(ls_manager=ls_manager, rng=self.rng)

        start_time = time.process_time()

        self._add_to_pool(current_routes)

        # 2. Outer Loop (Multi-start equivalents)
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and (time.process_time() - start_time) > self.params.time_limit:
                break

            iter_routes = copy.deepcopy(best_routes)

            # 3. Inner Loop (Iterated Local Search using RVND)
            for _ in range(self.params.ils_iterations):
                if self.params.time_limit > 0 and (time.process_time() - start_time) > self.params.time_limit:
                    break

                # Perturbation
                perturbed = self._perturb(iter_routes)

                # RVND Local Search
                ls_routes = rvnd.apply(perturbed)
                self._add_to_pool(ls_routes)

                ls_profit = self.calculate_profit(ls_routes)

                # Acceptance Criterion: Simple simulated annealing or strict descent
                # Here we apply strict improvement or minor worsening acceptance typical in strict ILS
                if ls_profit > current_profit - 1e-4:
                    iter_routes = copy.deepcopy(ls_routes)
                    current_profit = ls_profit

                if ls_profit > best_profit:
                    best_routes = copy.deepcopy(ls_routes)
                    best_profit = ls_profit

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                current_profit=current_profit,
                score=3 if ls_profit >= best_profit else 0,
            )

        # 4. Set Partitioning Refinement mapping Global Combinations
        if self.params.use_set_partitioning and self.route_pool:
            sp_routes, sp_profit, sp_cost = self.solve_set_partitioning()
            if sp_routes and sp_profit > best_profit:
                best_routes = sp_routes
                best_profit = sp_profit

        best_cost = self.calculate_cost(best_routes)
        return best_routes, best_profit, best_cost
