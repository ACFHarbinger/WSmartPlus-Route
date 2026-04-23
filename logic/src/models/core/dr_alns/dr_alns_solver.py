"""DR-ALNS Solver implementation.

This module provides the `DRALNSSolver`, which implements a Deep Reinforcement
Learning controlled Adaptive Large Neighborhood Search (AAAI 2024). It uses
a PPO agent to dynamically select destroy/repair operators and configure
meta-parameters (severity, temperature) during the search process.

Attributes:
    DRALNSSolver: Solver orchestrating PPO-based meta-heuristic control.

Example:
    >>> from logic.src.models.core.dr_alns.dr_alns_solver import DRALNSSolver
    >>> solver = DRALNSSolver(dist_matrix, wastes, capacity, R, C, agent)
    >>> best_routes, profit, cost = solver.solve()
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from logic.src.policies.helpers.operators import (
    cluster_removal,
    greedy_insertion,
    random_removal,
    regret_2_insertion,
    worst_removal,
)
from logic.src.policies.helpers.operators.solution_initialization import (
    build_nn_routes,
)

from .ppo_agent import DRALNSPPOAgent, DRALNSState


class DRALNSSolver:
    """Adaptive Large Neighborhood Search with PPO Control.

    The solver maintains a current solution and iteratively improves it by
    applying destruction and repair operators. Unlike standard ALNS which uses
    bandit-based weights, DR-ALNS uses a PPO agent that observes the search
    state to make decisions.

    Attributes:
        dist_matrix (np.ndarray): Symmetric or asymmetric distance matrix.
        wastes (Dict[int, float]): Mapping of customer nodes to waste inventory.
        capacity (float): Vehicle volume/weight constraint.
        R (float): Revenue multiplier per unit waste.
        C (float): Cost multiplier per unit distance.
        agent (DRALNSPPOAgent): Trained neural controller for operator selection.
        max_iterations (int): Search budget.
        mandatory_nodes (List[int]): Nodes that must be included in routes.
        device (torch.device): Hardware accelerator for agent inference.
        destroy_ops (List[Callable]): Registered destruction heuristics.
        repair_ops (List[Callable]): Registered reconstruction heuristics.
        state (DRALNSState): Search trajectory context tracker.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        agent: DRALNSPPOAgent,
        max_iterations: int = 100,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initializes the DR-ALNS solver.

        Args:
            dist_matrix: Distance matrix of shape [N+1, N+1].
            wastes: Node waste amounts.
            capacity: Vehicle capacity limit.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            agent: The PPO neural agent.
            max_iterations: Number of ALNS iterations to perform.
            mandatory_nodes: Set of nodes that must be visited.
            seed: Random seed for stochastic components.
            device: Computing device for the neural network.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.agent = agent
        self.max_iterations = max_iterations
        self.mandatory_nodes = mandatory_nodes or []
        self.device = device or torch.device("cpu")

        # Configure agent for inference
        self.agent.to(self.device)
        self.agent.eval()

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(seed)

        # Standard operator pool
        self.destroy_ops = [
            self._random_removal,
            self._worst_removal,
            self._cluster_removal,
        ]

        self.repair_ops = [
            self._greedy_insertion,
            self._regret_2_insertion,
        ]

        self.state = DRALNSState()

    def solve(self, initial_routes: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """Runs the PPO-controlled ALNS optimization loop.

        Args:
            initial_routes: Optional starting solution to refine.

        Returns:
            Tuple[List[List[int]], float, float]:
                - best_routes (List[List[int]]): The highest profit routes found.
                - best_profit (float): Total reward of the best routes.
                - best_cost (float): Total routing distance cost of best routes.
        """
        # 1. Initialize solution if none provided
        routes = self._build_initial_solution() if initial_routes is None else copy.deepcopy(initial_routes)

        current_profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = current_profit
        previous_profit = current_profit

        iterations_since_best = 0
        accepted = True
        new_best = False

        # 2. Iterative search
        for iteration in range(self.max_iterations):
            # Update agent context
            self.state.update(
                best_profit=best_profit,
                current_profit=current_profit,
                previous_profit=previous_profit,
                new_accepted=accepted,
                new_best=new_best,
                iteration=iteration,
                max_iterations=self.max_iterations,
                iterations_since_best=iterations_since_best,
            )

            state_tensor = self.state.to_tensor(self.device)

            # Neural action selection
            with torch.no_grad():
                actions, _, _ = self.agent.get_action(state_tensor, deterministic=True)

            destroy_idx = actions["destroy"]
            repair_idx = actions["repair"]
            severity_idx = actions["severity"]
            temp_idx = actions["temp"]

            # Map discrete actions to continuous parameters
            destroy_severity = (severity_idx + 1) / 10.0
            temperature = (temp_idx + 1) * 0.1

            # Operator application
            try:
                new_routes = self._apply_operators(routes, destroy_idx, repair_idx, destroy_severity)
                new_profit = self._evaluate(new_routes)
            except Exception:
                continue

            # Check acceptance (Simulated Annealing logic with agent-selected temp)
            accepted = self._accept_solution(current_profit, new_profit, temperature)

            if accepted:
                previous_profit = current_profit
                routes = new_routes
                current_profit = new_profit

                if current_profit > best_profit:
                    best_routes = copy.deepcopy(routes)
                    best_profit = current_profit
                    iterations_since_best = 0
                    new_best = True
                else:
                    iterations_since_best += 1
                    new_best = False
            else:
                iterations_since_best += 1
                new_best = False

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    def _apply_operators(
        self,
        routes: List[List[int]],
        destroy_idx: int,
        repair_idx: int,
        severity: float,
    ) -> List[List[int]]:
        """Applies high-level destroy and repair heuristics.

        Args:
            routes: Current solution paths.
            destroy_idx: Index of removal operator.
            repair_idx: Index of insertion operator.
            severity: Ratio of nodes to remove [0.1, 1.0].

        Returns:
            List[List[int]]: Reconstructed solution.
        """
        total_nodes = sum(len(route) for route in routes)
        n_remove = 0 if total_nodes == 0 else max(1, int(total_nodes * severity))

        destroy_op = self.destroy_ops[destroy_idx]
        partial_routes, removed_nodes = destroy_op(copy.deepcopy(routes), n_remove)

        repair_op = self.repair_ops[repair_idx]
        repaired_routes = repair_op(partial_routes, removed_nodes)

        return repaired_routes

    def _accept_solution(self, current_profit: float, new_profit: float, temperature: float) -> bool:
        """Determines if a candidate solution should be adopted.

        Args:
            current_profit: Objective value of incumbent.
            new_profit: Objective value of candidate.
            temperature: SA parameter controlled by PPO.

        Returns:
            bool: True if candidate is accepted.
        """
        if new_profit >= current_profit:
            return True

        if temperature <= 0:
            return False

        delta = new_profit - current_profit
        probability = np.exp(delta / temperature)
        return self.random.random() < probability

    def _random_removal(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Removes `n` nodes uniformly at random."""
        return random_removal(routes, n, rng=self.random)

    def _worst_removal(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Removes `n` nodes that contribute most to total cost."""
        return worst_removal(routes, n, self.dist_matrix)

    def _cluster_removal(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Removes a cluster of spatially proximate nodes."""
        return cluster_removal(routes, n, self.dist_matrix, self.nodes, rng=self.random)

    def _greedy_insertion(self, partial_routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Inserts removed nodes into best possible positions sequentially."""
        return greedy_insertion(
            partial_routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _regret_2_insertion(self, partial_routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Inserts nodes based on the difference between best and second best positions."""
        return regret_2_insertion(
            partial_routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _build_initial_solution(self) -> List[List[int]]:
        """Generates a starting solution using nearest neighbor logic."""
        return build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Computes the business objective: Revenue - Cost.

        Args:
            routes: solution path list.

        Returns:
            float: net profit.
        """
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(node, 0.0) * self.R for route in routes for node in route)
        cost = self._cost(routes) * self.C
        return revenue - cost

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculates total accumulated distance of all routes.

        Args:
            routes: solution path list.

        Returns:
            float: total distance.
        """
        total_cost = 0.0
        for route in routes:
            if not route:
                continue
            total_cost += self.dist_matrix[0, route[0]]
            for i in range(len(route) - 1):
                total_cost += self.dist_matrix[route[i], route[i + 1]]
            total_cost += self.dist_matrix[route[-1], 0]
        return total_cost
