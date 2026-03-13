"""
Distance-Based Particle Swarm Optimization for VRPP.

Standard PSO where attraction to the global best solution decays exponentially
with solution distance. Replaces the metaphor-based "Firefly Algorithm":
- "Fireflies" → Particles
- "Light intensity" → Objective function value (fitness)
- "Attractiveness" → Distance-weighted attraction to global best
- "Random walk" → Exploration operator

Algorithm:
    1. Initialize swarm of particles (solutions)
    2. For each iteration:
        a. For each particle pair (i, j where fitness[j] > fitness[i]):
            - Compute Hamming distance d between solutions
            - With probability β(d) = β₀ × exp(-γ × d²), move particle i toward j
        b. With probability α, apply random walk exploration
        c. Update global best solution

Complexity:
    - Time: O(T × N² × n²) where T = iterations, N = pop_size, n = nodes
    - Space: O(N × n) for swarm storage
    - Distance computation: O(n) per pair (edge set symmetric difference)

Reference:
    Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
    Proceedings of ICNN'95 - International Conference on Neural Networks.
    Ai, T. J., & Kachitvichyanukul, V. (2009). "A particle swarm optimization
    for the vehicle routing problem with simultaneous pickup and delivery"
"""

import contextlib
import copy
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..other.operators import greedy_insertion, random_removal
from .params import DistancePSOParams


class DistancePSOSolver(PolicyVizMixin):
    """
    Distance-Based Particle Swarm Optimization solver for VRPP.

    Each particle represents a routing solution. Particles are attracted
    to better solutions with strength decaying exponentially by Hamming distance.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: DistancePSOParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Distance-Based PSO solver.

        Args:
            dist_matrix: Distance matrix [n+1 × n+1] including depot at index 0.
            wastes: Dictionary mapping node IDs to waste quantities.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            params: PSO configuration parameters.
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

        # Initialize Local Search once for reuse
        from ..ant_colony_optimization.k_sparse_aco.params import ACOParams
        from ..other.local_search.local_search_aco import ACOLocalSearch

        aco_params = ACOParams(local_search_iterations=self.params.local_search_iterations)
        self.ls = ACOLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=aco_params,
            seed=seed,
        )

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute Distance-Based Particle Swarm Optimization.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initialize swarm (particle population)
        swarm = [self._initialize_particle() for _ in range(self.params.population_size)]
        fitness_values = [self._evaluate(particle) for particle in swarm]

        # Track global best
        best_idx = int(np.argmax(fitness_values))
        best_routes = copy.deepcopy(swarm[best_idx])
        best_profit = fitness_values[best_idx]
        best_cost = self._cost(best_routes)

        # PSO main loop
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Pairwise attraction: Each particle moves toward better particles
            for i in range(self.params.population_size):
                moved = False

                for j in range(self.params.population_size):
                    if fitness_values[j] <= fitness_values[i]:
                        continue

                    # Compute Hamming distance between particles
                    hamming_distance = self._hamming_distance(swarm[i], swarm[j])

                    # Distance-dependent attraction weight
                    attraction_weight = self.params.initial_attraction * np.exp(
                        -self.params.distance_decay * hamming_distance * hamming_distance
                    )

                    # Probabilistically move toward better particle
                    if self.random.random() < attraction_weight:
                        new_particle = self._attract_toward(swarm[i], swarm[j])
                        new_fitness = self._evaluate(new_particle)
                        if new_fitness > fitness_values[i]:
                            swarm[i] = new_particle
                            fitness_values[i] = new_fitness
                            moved = True

                # Random walk exploration (with probability α)
                if not moved or self.random.random() < self.params.exploration_rate:
                    explored = self._random_walk(swarm[i])
                    explored_fitness = self._evaluate(explored)
                    if explored_fitness > fitness_values[i]:
                        swarm[i] = explored
                        fitness_values[i] = explored_fitness

                # Update global best
                if fitness_values[i] > best_profit:
                    best_routes = copy.deepcopy(swarm[i])
                    best_profit = fitness_values[i]
                    best_cost = self._cost(best_routes)

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                population_size=self.params.population_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Particle Initialization
    # ------------------------------------------------------------------

    def _initialize_particle(self) -> List[List[int]]:
        """
        Initialize a single particle using nearest-neighbor heuristic.

        Randomized node ordering creates diverse initial swarm distribution.

        Returns:
            A feasible routing solution as list of routes.

        Complexity: O(n²) for nearest-neighbor construction.
        """
        from logic.src.policies.other.operators.heuristics.nn_initialization import build_nn_routes

        routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.random,
        )
        return routes

    # ------------------------------------------------------------------
    # Distance Metrics
    # ------------------------------------------------------------------

    def _hamming_distance(self, routes_a: List[List[int]], routes_b: List[List[int]]) -> int:
        """
        Compute Hamming distance between two routing solutions.

        Defined as the number of edges present in one solution but not the other
        (symmetric edge set difference). This measures structural dissimilarity.

        Args:
            routes_a: First routing solution.
            routes_b: Second routing solution.

        Returns:
            Integer Hamming distance ≥ 0.

        Complexity: O(n) for edge set construction and comparison.
        """
        edges_a = self._extract_edge_set(routes_a)
        edges_b = self._extract_edge_set(routes_b)
        return len(edges_a.symmetric_difference(edges_b))

    @staticmethod
    def _extract_edge_set(routes: List[List[int]]) -> Set[Tuple[int, int]]:
        """
        Extract the set of directed edges from a routing solution.

        Includes depot→first_node, inter-node, and last_node→depot edges.

        Args:
            routes: List of routes.

        Returns:
            Set of directed edges (tuples).

        Complexity: O(n) for route traversal.
        """
        edges = set()
        for route in routes:
            if not route:
                continue
            edges.add((0, route[0]))
            for k in range(len(route) - 1):
                edges.add((route[k], route[k + 1]))
            edges.add((route[-1], 0))
        return edges

    # ------------------------------------------------------------------
    # Particle Movement Operators
    # ------------------------------------------------------------------

    def _attract_toward(self, current_particle: List[List[int]], target_particle: List[List[int]]) -> List[List[int]]:
        """
        Move current particle toward target particle via guided node insertion.

        Extracts nodes that are in target but not in current, scores each by
        profitability, and inserts them greedily. This implements the PSO
        "velocity update" in discrete space.

        Scoring function per candidate node n:
            score(n) = α_profit × profit(n) + β_will × fill_level(n) - γ_cost × insertion_cost(n)

        Args:
            current_particle: Current particle (routing solution).
            target_particle: Target particle with higher fitness.

        Returns:
            Updated particle after attraction operation.

        Complexity: O(n²) for insertion cost computation + greedy insertion.
        """
        current_nodes = {n for route in current_particle for n in route}
        target_nodes = {n for route in target_particle for n in route}
        candidate_nodes = [n for n in target_nodes if n not in current_nodes]

        if not candidate_nodes:
            return copy.deepcopy(current_particle)

        # Score candidates by multi-objective function
        scored_candidates = []
        for node in candidate_nodes:
            profit = self.wastes.get(node, 0.0) * self.R
            fill_level = self.wastes.get(node, 0.0)  # Proxy for urgency
            insertion_cost = self._compute_best_insertion_cost(node, current_particle)
            score = (
                self.params.alpha_profit * profit
                + self.params.beta_will * fill_level
                - self.params.gamma_cost * insertion_cost
            )
            scored_candidates.append((score, node))

        scored_candidates.sort(reverse=True)
        selected_nodes = [node for _, node in scored_candidates]

        # Insert selected nodes greedily
        routes = copy.deepcopy(current_particle)
        with contextlib.suppress(Exception):
            routes = greedy_insertion(
                routes,
                selected_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            # Apply local search refinement
            return self.ls.optimize(routes)
        return routes

    def _compute_best_insertion_cost(self, node: int, routes: List[List[int]]) -> float:
        """
        Compute minimum insertion cost for a node into existing routes.

        Finds the position that minimizes additional distance when inserting node.

        Args:
            node: Node index to insert.
            routes: Existing routes.

        Returns:
            Minimum additional distance required.

        Complexity: O(n) for trying all insertion positions.
        """
        min_cost = float("inf")
        for route in routes:
            for i in range(len(route) + 1):
                prev = 0 if i == 0 else route[i - 1]
                nxt = 0 if i == len(route) else route[i]
                cost = self.dist_matrix[prev][node] + self.dist_matrix[node][nxt] - self.dist_matrix[prev][nxt]
                if cost < min_cost:
                    min_cost = cost

        # If no routes exist, cost = depot → node → depot
        if not routes:
            min_cost = self.dist_matrix[0][node] + self.dist_matrix[node][0]

        return max(0.0, min_cost)

    def _random_walk(self, particle: List[List[int]]) -> List[List[int]]:
        """
        Apply random walk exploration to particle.

        Removes random nodes and reinserts them greedily. This maintains
        swarm diversity and prevents premature convergence.

        Args:
            particle: Current particle (routing solution).

        Returns:
            Perturbed particle after random walk.

        Complexity: O(n²) for destroy-repair operation.
        """
        try:
            n_remove = max(3, self.params.n_removal)
            partial_routes, removed_nodes = random_removal(particle, n_remove, self.random)
            repaired_routes = greedy_insertion(
                partial_routes,
                removed_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            # Apply local search refinement
            return self.ls.optimize(repaired_routes)
        except Exception:
            return copy.deepcopy(particle)

    # ------------------------------------------------------------------
    # Fitness Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Evaluate particle fitness (net profit).

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
