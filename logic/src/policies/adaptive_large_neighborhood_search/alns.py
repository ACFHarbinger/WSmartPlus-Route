"""
Adaptive Large Neighborhood Search (ALNS) policy module.

This module provides the main entry points for the ALNS metaheuristic,
dispatching to specialized implementations based on configuration.

Reference:
    Pisinger, D., & Ropke, S. "An Adaptive Large Neighborhood Search
    Heuristic for the Pickup and Delivery Problem with Time Windows.", 2005.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder
from logic.src.utils.functions import safe_exp

from ..other.operators import (
    cluster_removal,
    greedy_insertion,
    random_removal,
    regret_2_insertion,
    worst_removal,
)
from .params import ALNSParams


class ALNSSolver:
    """
    Custom implementation of Adaptive Large Neighborhood Search for CVRP.
    Follows Pisinger & Ropke (2007) with segment-based weight updates and noise.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ALNSParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
        recorder: Optional[PolicyStateRecorder] = None,
    ):
        """
        Initialize the ALNS solver.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(seed) if seed is not None else random.Random(42)

        if recorder is not None:
            self._viz_record = recorder.record

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Operator registry
        self.destroy_ops = [
            lambda r, n: random_removal(r, n, rng=self.random),
            lambda r, n: worst_removal(r, n, self.dist_matrix),
            lambda r, n: cluster_removal(r, n, self.dist_matrix, self.nodes, rng=self.random),
        ]

        # Repair with Noise (Pisinger & Ropke, 2007)
        noise_factor = 0.1
        self.repair_ops = [
            lambda r, n: greedy_insertion(
                r,
                n,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=False,
                noise=(self.random.uniform(-noise_factor, noise_factor) if self.random.random() < 0.5 else 0.0),
            ),
            lambda r, n: regret_2_insertion(
                r,
                n,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=False,
                noise=(self.random.uniform(-noise_factor, noise_factor) if self.random.random() < 0.5 else 0.0),
            ),
        ]

        # Segment-based weight update logic
        self.segment_size = 100
        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)
        self.destroy_scores = [0.0] * len(self.destroy_ops)
        self.repair_scores = [0.0] * len(self.repair_ops)
        self.destroy_counts = [0] * len(self.destroy_ops)
        self.repair_counts = [0] * len(self.repair_ops)
        self.lambda_decay = 0.8

    def _initialize_solve(self, initial_solution: Optional[List[List[int]]]):
        current_routes = initial_solution or self.build_initial_solution()
        best_routes = copy.deepcopy(current_routes)
        best_cost = self.calculate_cost(best_routes)
        best_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in best_routes for node_idx in route)
        best_profit = best_rev - best_cost
        return current_routes, best_routes, best_profit, best_cost

    def _select_and_apply_operators(self, current_routes):
        d_idx = self.select_operator(self.destroy_weights)
        r_idx = self.select_operator(self.repair_weights)
        destroy_op = self.destroy_ops[d_idx]
        repair_op = self.repair_ops[r_idx]

        current_n_nodes = sum(len(route) for route in current_routes)
        if current_n_nodes == 0:
            n_remove = 0
        else:
            lower_bound = min(current_n_nodes, 2)
            max_pct_remove = int(current_n_nodes * self.params.max_removal_pct)
            upper_bound = min(current_n_nodes, max(lower_bound + 1, max_pct_remove))
            n_remove = self.random.randint(lower_bound, upper_bound)

        new_routes, removed = destroy_op(copy.deepcopy(current_routes), n_remove)
        new_routes = repair_op(new_routes, removed)
        return new_routes, d_idx, r_idx

    def _accept_solution(self, current_profit, new_profit, T):
        delta = current_profit - new_profit
        if delta < -1e-6:
            return True
        prob = safe_exp(-delta / T) if T > 0 else 0
        return self.random.random() < prob

    def _update_weights(self, d_idx, r_idx, score):
        self.destroy_scores[d_idx] += score
        self.repair_scores[r_idx] += score
        self.destroy_counts[d_idx] += 1
        self.repair_counts[r_idx] += 1

    def _end_segment(self):
        for i in range(len(self.destroy_weights)):
            if self.destroy_counts[i] > 0:
                avg_score = self.destroy_scores[i] / self.destroy_counts[i]
                self.destroy_weights[i] = (
                    self.lambda_decay * self.destroy_weights[i] + (1 - self.lambda_decay) * avg_score
                )
            self.destroy_scores[i] = 0.0
            self.destroy_counts[i] = 0
        for i in range(len(self.repair_weights)):
            if self.repair_counts[i] > 0:
                avg_score = self.repair_scores[i] / self.repair_counts[i]
                self.repair_weights[i] = (
                    self.lambda_decay * self.repair_weights[i] + (1 - self.lambda_decay) * avg_score
                )
            self.repair_scores[i] = 0.0
            self.repair_counts[i] = 0

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        start_time = time.process_time()
        current_routes, best_routes, best_profit, best_cost = self._initialize_solve(initial_solution)
        current_profit = best_profit
        T = self.params.start_temp

        for _it in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            new_routes, d_idx, r_idx = self._select_and_apply_operators(current_routes)
            new_cost = self.calculate_cost(new_routes)
            new_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in new_routes for node_idx in route)
            new_profit = new_rev - new_cost

            accept = self._accept_solution(current_profit, new_profit, T)
            # Pisinger & Ropke (2007) score rewards:
            # σ1: New global best found
            # σ2: Better than current
            # σ3: Accepted worse
            score = 0
            if accept:
                if new_profit > best_profit + 1e-6:
                    best_routes = copy.deepcopy(new_routes)
                    best_profit = new_profit
                    best_cost = new_cost
                    score = 15  # σ1
                elif new_profit > current_profit + 1e-6:
                    score = 9  # σ2
                else:
                    score = 2  # σ3
                current_routes = new_routes
                current_profit = new_profit
            else:
                score = 0

            self._update_weights(d_idx, r_idx, score)
            if (_it + 1) % self.segment_size == 0:
                self._end_segment()

            # Cooling schedule
            T *= self.params.cooling_rate

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=_it,
                d_idx=d_idx,
                r_idx=r_idx,
                best_profit=best_profit,
                current_profit=current_profit,
                temperature=T,
                accepted=int(accept),
                score=score,
            )
        return best_routes, best_profit, best_cost

    def select_operator(self, weights: List[float]) -> int:
        total = sum(weights)
        r = self.random.uniform(0, total)
        curr = 0.0
        for i, w in enumerate(weights):
            curr += w
            if curr >= r:
                return i
        return len(weights) - 1

    def calculate_cost(self, routes: List[List[int]]) -> float:
        total_dist = 0
        for route in routes:
            if not route:
                continue
            dist = self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                dist += self.dist_matrix[route[i]][route[i + 1]]
            dist += self.dist_matrix[route[-1]][0]
            total_dist += dist
        return total_dist * self.C

    def build_initial_solution(self) -> List[List[int]]:
        nodes = self.nodes[:]
        self.random.shuffle(nodes)
        routes, curr_route, load = [], [], 0.0
        mandatory_set = set(self.mandatory_nodes) if self.mandatory_nodes else set()
        for node in nodes:
            waste = self.wastes.get(node, 0)
            revenue = waste * self.R
            if node not in mandatory_set and revenue < (self.dist_matrix[0][node] + self.dist_matrix[node][0]) * self.C:
                continue
            if waste > self.capacity:
                continue
            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    routes.append(curr_route)
                curr_route, load = [node], waste
        if curr_route:
            routes.append(curr_route)
        return routes
