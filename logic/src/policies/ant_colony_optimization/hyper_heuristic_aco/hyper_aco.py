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
"""

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
        demands: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Optional[HyperACOParams] = None,
    ):
        self.dist_matrix = np.array(dist_matrix)
        self.demands = demands
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params or HyperACOParams()

        # Operator configuration
        self.operators = [op for op in self.params.operators if op in HYPER_OPERATORS]
        self.n_operators = len(self.operators)
        self.op_to_idx = {op: i for i, op in enumerate(self.operators)}
        self.idx_to_op = {i: op for i, op in enumerate(self.operators)}

        # Pheromone matrix: tau[i][j] = pheromone for transition i -> j
        # Include dummy "start" node at index n_operators
        self._init_pheromones()

        # Heuristic information (operator success rates)
        self.eta = np.ones(self.n_operators)  # Initially uniform
        self.success_counts = np.ones(self.n_operators)
        self.use_counts = np.ones(self.n_operators)

        # Best solution tracking
        self.best_routes: List[List[int]] = []
        self.best_cost = float("inf")
        self.best_sequence: List[str] = []

    def _init_pheromones(self):
        """Initialize pheromone matrix."""
        size = self.n_operators + 1  # +1 for start node
        self.tau = np.full((size, size), self.params.tau_0)

    def solve(self, initial_routes: List[List[int]]) -> Tuple[List[List[int]], float, float]:
        """
        Solve using Hyper-Heuristic ACO.

        Args:
            initial_routes: Starting solution to improve.

        Returns:
            Tuple[List[List[int]], float, float]: (best_routes, best_profit, best_cost)
        """
        self.best_routes = [r[:] for r in initial_routes]
        self.best_cost = self._calculate_cost(self.best_routes)

        t_start = time.time()

        for iteration in range(self.params.max_iterations):
            if time.time() - t_start > self.params.time_limit:
                break

            # Generate solutions for all ants
            ant_solutions: List[Tuple[List[List[int]], float, List[str]]] = []

            for _ in range(self.params.n_ants):
                # Construct operator sequence
                sequence = self._construct_sequence()

                # Apply sequence to a copy of current best
                routes = [r[:] for r in self.best_routes]
                routes = self._apply_sequence(routes, sequence)

                cost = self._calculate_cost(routes)
                ant_solutions.append((routes, cost, sequence))

            # Find iteration best
            ant_solutions.sort(key=lambda x: x[1])
            iter_routes, iter_cost, iter_sequence = ant_solutions[0]

            # Update best if improved
            if iter_cost < self.best_cost:
                self.best_routes = [r[:] for r in iter_routes]
                self.best_cost = iter_cost
                self.best_sequence = iter_sequence[:]

            # Pheromone update
            self._evaporate_pheromones()
            self._deposit_pheromones(ant_solutions)
            self._update_heuristics()

        # Final profit calculation
        collected_revenue = sum(self.demands.get(node, 0) * self.R for route in self.best_routes for node in route)
        best_profit = collected_revenue - self.best_cost * self.C

        return self.best_routes, best_profit, self.best_cost

    def _construct_sequence(self) -> List[str]:
        """
        Construct an operator sequence using ACO state transition rule.
        """
        sequence = []
        current = self.n_operators  # Start node

        for _ in range(self.params.sequence_length):
            next_op = self._select_next_operator(current)
            sequence.append(self.idx_to_op[next_op])
            current = next_op

        return sequence

    def _select_next_operator(self, current: int) -> int:
        """
        Select next operator using pseudo-random proportional rule.
        """
        # Calculate transition probabilities
        probs = np.zeros(self.n_operators)
        for j in range(self.n_operators):
            probs[j] = (self.tau[current, j] ** self.params.alpha) * (self.eta[j] ** self.params.beta)

        total = probs.sum()
        if total == 0:
            return random.randint(0, self.n_operators - 1)

        probs /= total

        # Pseudo-random proportional rule
        if random.random() < self.params.q0:
            # Exploitation: choose best
            return int(np.argmax(probs))
        else:
            # Exploration: roulette wheel
            return int(np.random.choice(self.n_operators, p=probs))

    def _apply_sequence(self, routes: List[List[int]], sequence: List[str]) -> List[List[int]]:
        """
        Apply a sequence of operators to the routes.
        """
        ctx = HyperOperatorContext(
            routes=routes,
            dist_matrix=self.dist_matrix,
            demands=self.demands,
            capacity=self.capacity,
            C=self.C,
        )

        for op_name in sequence:
            op_func = HYPER_OPERATORS.get(op_name)
            if op_func:
                improved = op_func(ctx)
                op_idx = self.op_to_idx.get(op_name, 0)
                self.use_counts[op_idx] += 1
                if improved:
                    self.success_counts[op_idx] += 1

        return ctx.routes

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

    def _deposit_pheromones(self, solutions: List[Tuple[List[List[int]], float, List[str]]]):
        """
        Deposit pheromones based on solution quality.
        Uses elitist strategy: only best solutions deposit.
        """
        if not solutions:
            return

        # Best solution deposits more
        best_routes, best_cost, best_seq = solutions[0]
        if best_cost > 0:
            delta = 1.0 / best_cost

            prev = self.n_operators  # Start node
            for op_name in best_seq:
                op_idx = self.op_to_idx.get(op_name, 0)
                self.tau[prev, op_idx] += delta
                prev = op_idx

        np.clip(self.tau, self.params.tau_min, self.params.tau_max, out=self.tau)

    def _update_heuristics(self):
        """Update heuristic information based on operator success rates."""
        self.eta = self.success_counts / self.use_counts
        self.eta = np.clip(self.eta, 0.01, 10.0)
