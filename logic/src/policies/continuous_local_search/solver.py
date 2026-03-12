"""
Continuous Local Search with Trigonometric Perturbations for VRPP.

Gradient-free continuous optimization using trigonometric step functions.
Replaces the metaphor-based "Sine Cosine Algorithm":
- "Position vectors" → Continuous solution encoding
- "Destination point" → Best solution (global attractor)
- "Sine/Cosine update" → Directional perturbation operators
- "Parameter a" → Adaptive step size (exploration → exploitation)

Algorithm:
    1. Initialize population in continuous space (real-valued vectors)
    2. For each iteration:
        a. Update step size α = α_max × (1 - t/T) (linear decay)
        b. For each solution vector:
            - Compute perturbation direction toward global best
            - Apply trigonometric step: sin(θ) or cos(θ) with probability 0.5
            - Update position: x' = x + α × trig(θ) × |β × x_best - x|
            - Decode to discrete solution and evaluate fitness
        c. Track global best solution

Complexity:
    - Time: O(T × N × n²) where T = iterations, N = pop_size, n = nodes
    - Space: O(N × n) for continuous population + O(N × routes) for discrete solutions
    - Decoding: O(n²) per solution (greedy insertion + sigmoid binarization)

Reference:
    Mirjalili, S. "SCA: A Sine Cosine Algorithm for solving
    optimization problems.", 2016, Knowledge-Based Systems.
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from .params import ContinuousLocalSearchParams


class ContinuousLocalSearchSolver(PolicyVizMixin):
    """
    Continuous Local Search solver for VRPP using trigonometric perturbations.

    Maintains a population of continuous solution vectors and applies
    trigonometric step functions for exploration/exploitation balance.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ContinuousLocalSearchParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Continuous Local Search solver.

        Args:
            dist_matrix: Distance matrix [n+1 × n+1] including depot at index 0.
            wastes: Dictionary mapping node IDs to waste quantities.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            params: CLS configuration parameters.
            mandatory_nodes: Nodes that must be visited.
            seed: Random seed for reproducibility.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(seed) if seed is not None else random.Random()
        self.np_rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute Continuous Local Search with trigonometric perturbations.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()
        T = self.params.max_iterations

        # Initialize population in continuous space [-1, 1]^n
        continuous_population = self.np_rng.uniform(-1.0, 1.0, (self.params.population_size, self.n_nodes))

        # Decode to discrete solutions and evaluate
        discrete_population = [self._decode_to_routes(x) for x in continuous_population]
        fitness_values = [self._evaluate(routes) for routes in discrete_population]

        # Track global best
        best_idx = int(np.argmax(fitness_values))
        best_continuous = continuous_population[best_idx].copy()
        best_routes = copy.deepcopy(discrete_population[best_idx])
        best_profit = fitness_values[best_idx]
        best_cost = self._cost(best_routes)

        # Local search loop
        for t in range(T):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Adaptive step size: Linear decay from α_max to 0
            step_size = self.params.max_step_size * (1.0 - t / T)

            # Update each solution vector
            for i in range(self.params.population_size):
                # Random parameters for perturbation
                r1 = self.random.uniform(0, step_size)  # Step magnitude
                r2 = self.random.uniform(0, 2 * math.pi)  # Angle parameter
                r3 = self.random.uniform(0, 2)  # Distance scaling
                r4 = self.random.random()  # Direction selector

                # Compute perturbation direction toward global best
                difference = r3 * best_continuous - continuous_population[i]

                # Apply trigonometric perturbation (sine or cosine)
                if r4 < 0.5:
                    continuous_population[i] = continuous_population[i] + r1 * math.sin(r2) * np.abs(difference)
                else:
                    continuous_population[i] = continuous_population[i] + r1 * math.cos(r2) * np.abs(difference)

                # Decode and evaluate
                discrete_population[i] = self._decode_to_routes(continuous_population[i])
                fitness_values[i] = self._evaluate(discrete_population[i])

                # Update global best
                if fitness_values[i] > best_profit:
                    best_continuous = continuous_population[i].copy()
                    best_routes = copy.deepcopy(discrete_population[i])
                    best_profit = fitness_values[i]
                    best_cost = self._cost(best_routes)

            self._viz_record(
                iteration=t,
                best_profit=best_profit,
                best_cost=best_cost,
                step_size=step_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Encoding/Decoding (Continuous ↔ Discrete)
    # ------------------------------------------------------------------

    def _decode_to_routes(self, continuous_vector: np.ndarray) -> List[List[int]]:
        """
        Decode continuous vector to discrete routing solution.

        Decoding Strategy:
            1. Apply sigmoid binarization: b[j] = 1 if σ(x[j]) > 0.5
            2. Always include mandatory nodes
            3. For optional nodes, include if sigmoid > 0.5 and capacity allows
            4. Sort selected nodes by continuous value (Largest Rank Value rule)
            5. Construct routes via greedy insertion

        Args:
            continuous_vector: Continuous solution encoding of length n.

        Returns:
            Discrete routing solution as list of routes.

        Complexity: O(n log n) for sorting + O(n²) for greedy insertion.
        """
        # Sigmoid activation for binarization (clipping for numerical stability)
        sigmoid_values = 1.0 / (1.0 + np.exp(-np.clip(continuous_vector, -100, 100)))

        mandatory_set = set(self.mandatory_nodes)
        selected_nodes: List[int] = []

        # Always include mandatory nodes
        for node in self.nodes:
            if node in mandatory_set:
                selected_nodes.append(node)

        # Select optional nodes by sigmoid threshold and capacity
        total_load = sum(self.wastes.get(n, 0.0) for n in selected_nodes)

        # Sort optional nodes by sigmoid value (descending) for Largest Rank Value decoding
        optional_nodes_sorted = sorted(
            [
                (sigmoid_values[idx], self.nodes[idx])
                for idx in range(self.n_nodes)
                if self.nodes[idx] not in mandatory_set
            ],
            reverse=True,
        )

        for sigmoid_val, node in optional_nodes_sorted:
            waste = self.wastes.get(node, 0.0)
            if sigmoid_val > 0.5 and total_load + waste <= self.capacity:
                selected_nodes.append(node)
                total_load += waste

        if not selected_nodes:
            return []

        # Construct routes via greedy insertion
        from logic.src.policies.other.operators.repair.greedy import greedy_insertion

        routes = greedy_insertion(
            [],
            selected_nodes,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=False,
        )

        return routes

    # ------------------------------------------------------------------
    # Fitness Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Evaluate solution fitness (net profit).

        Fitness = Revenue - Cost × C
        Revenue = Σ(waste_collected × R)
        Cost = Total distance traveled

        Args:
            routes: Routing solution.

        Returns:
            Net profit (higher is better).

        Complexity: O(n) for route traversal.
        """
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(n, 0.0) * self.R for route in routes for n in route)
        return revenue - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing distance.

        Args:
            routes: List of routes.

        Returns:
            Total distance traveled.

        Complexity: O(n) for route traversal.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
