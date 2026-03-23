"""
Hyper-Heuristic Ant Colony Optimization (Hyper-ACO).

This module implements a Hyper-Heuristic ACO where ants construct
sequences of local search operators rather than node sequences.
The pheromone matrix encodes transition probabilities between operators.

Key Features:
- Operator graph: Nodes represent local search operators (2-opt, swap, etc.)
- Pheromone trail: tau[i][j] = favorability of applying operator j after operator i
- Heuristic information: Dynamic based on operator success rate
- Solution evaluation: Apply operator sequence to a base solution

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization_hyper_heuristic import HyperHeuristicACO
    >>> solver = HyperHeuristicACO(dist_matrix, wastes, capacity, R, C, params)
    >>> best_solution = solver.solve(initial_solution)
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hyper_operators import (
    HYPER_OPERATORS,
    HyperOperatorContext,
)
from .params import HyperACOParams


class HyperHeuristicACO:
    """
    Hyper-Heuristic ACO solver.

    Ants construct sequences of local search operators.
    The best sequences are reinforced via pheromone updates.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Optional[HyperACOParams] = None,
        initial_solution: Optional[List[List[int]]] = None,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize HyperHeuristicACO.

        Args:
            dist_matrix: Distance matrix between nodes.
            wastes: Dictionary mapping node indices to wastes.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Hyper-heuristics parameters object.
            initial_solution: Optional starting routes.
            mandatory_nodes: List of mandatory node indices.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params or HyperACOParams()
        self.initial_solution = initial_solution or []
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(self.params.seed) if self.params.seed is not None else random.Random(42)
        self.np_rng = (
            np.random.default_rng(self.params.seed) if self.params.seed is not None else np.random.default_rng(42)
        )

        self.operator_names = list(HYPER_OPERATORS.keys())
        self.n_operators = len(self.operator_names)
        self.op_to_idx = {name: i for i, name in enumerate(self.operator_names)}

        # Pheromone matrix: n_operators x n_operators
        # tau[i][j] where i is the previous operator and j is the next.
        # We add one extra row for the start state (0).
        self.tau = np.full((self.n_operators + 1, self.n_operators), self.params.tau_0)

        # Heuristic information: derived from success rates
        self.eta = np.ones(self.n_operators)
        self.success_counts = np.zeros(self.n_operators)
        self.use_counts = np.ones(self.n_operators)

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Solve VRP using Hyper-ACO.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
        """
        best_routes = copy.deepcopy(self.initial_solution)
        best_cost = self._calculate_cost(best_routes)
        best_profit = sum(self.wastes.get(n, 0) * self.R for r in best_routes for n in r) - best_cost

        start_time = time.process_time()
        for _it in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # 1. Ant solutions construction
            ant_results: List[Tuple[List[List[int]], float, List[str]]] = []
            for _ant in range(self.params.n_ants):
                routes, sequence = self.build_solution()
                cost = self._calculate_cost(routes)
                ant_results.append((routes, cost, sequence))

            # 2. Identify iteration best
            ant_results.sort(key=lambda x: x[1])  # Sort by cost
            iter_best_routes, iter_best_cost, iter_best_sequence = ant_results[0]
            iter_best_profit = sum(self.wastes.get(n, 0) * self.R for r in iter_best_routes for n in r) - iter_best_cost

            if iter_best_profit > best_profit:
                best_profit = iter_best_profit
                best_cost = iter_best_cost
                best_routes = copy.deepcopy(iter_best_routes)

            # 3. Pheromone updates
            self._evaporate_pheromones()

            # Pheromone deposit: tau[i][j] += Q / cost for the iteration-best sequence
            current_op_idx = self.n_operators  # Start state
            for op_name in iter_best_sequence:
                next_op_idx = self.op_to_idx[op_name]
                self.tau[current_op_idx][next_op_idx] += self.params.Q / max(iter_best_cost, 1e-9)
                current_op_idx = next_op_idx

            # 4. Success-based Heuristics & Meta-data updates
            self._update_heuristics()

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=_it,
                best_profit=best_profit,
                best_cost=best_cost,
                iter_best_cost=iter_best_cost,
                tau_mean=float(self.tau.mean()),
                eta_mean=float(self.eta.mean()),
            )

        collected_rev = sum(self.wastes.get(n, 0) * self.R for r in best_routes for n in r)
        final_cost = best_cost / self.C if self.C > 0 else best_cost
        return best_routes, collected_rev - best_cost, final_cost

    def build_solution(self) -> Tuple[List[List[int]], List[str]]:
        """
        Build a solution by applying a sequence of operators.

        Returns:
            Tuple[List[List[int]], List[str]]: Modified routes and the operator sequence.
        """
        sequence = self._select_sequence()
        ctx = HyperOperatorContext(
            routes=copy.deepcopy(self.initial_solution),
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            profit_aware_operators=self.params.profit_aware_operators,
            vrpp=self.params.vrpp,
        )

        for op_name in sequence:
            op_func = HYPER_OPERATORS.get(op_name)
            if op_func:
                improved = op_func(ctx)
                op_idx = self.op_to_idx.get(op_name, 0)
                self.use_counts[op_idx] += 1
                if improved:
                    self.success_counts[op_idx] += 1

        return ctx.routes, sequence

    def _select_sequence(self) -> List[str]:
        """Construct an operator sequence using ACO rules."""
        sequence = []
        current_op_idx = self.n_operators  # Start state (index n_operators in tau)

        for _ in range(self.params.sequence_length):
            # Proportional selection logic (tau[prev][next] * eta[next])
            numerator = (self.tau[current_op_idx] ** self.params.alpha) * (self.eta**self.params.beta)
            denom = np.sum(numerator)
            probs = numerator / denom if denom > 1e-12 else np.ones(self.n_operators) / self.n_operators

            next_op_idx = self.np_rng.choice(self.n_operators, p=probs)
            sequence.append(self.operator_names[next_op_idx])
            current_op_idx = next_op_idx

        return sequence

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0, route[0]]
            for i in range(len(route) - 1):
                total += self.dist_matrix[route[i], route[i + 1]]
            total += self.dist_matrix[route[-1], 0]
        return total * self.C

    def _evaporate_pheromones(self):
        """Apply pheromone evaporation."""
        self.tau *= 1 - self.params.rho
        np.clip(self.tau, self.params.tau_min, self.params.tau_max, out=self.tau)

    def _update_heuristics(self):
        """Update heuristic information based on operator success rates."""
        # eta = success rate with a floor
        self.eta = self.success_counts / self.use_counts
        self.eta = np.clip(self.eta, 0.01, 10.0)
