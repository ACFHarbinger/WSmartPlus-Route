"""
(μ+λ) Evolution Strategy for VRPP.

Canonical evolution strategy with population-based search. Replaces the
metaphor-based "Harmony Search" with proper mathematical terminology:
- "Harmony Memory" → Population/Archive
- "Improvisation" → Offspring generation via recombination + mutation
- "HMCR" → Recombination rate
- "Pitch Adjustment" → Mutation operator
- "Offspring size" → λ (number of candidates generated per cycle)

Algorithm:
    1. Initialize population of μ solutions
    2. For each iteration:
        a. Create λ offspring via recombination + mutation
        b. Evaluate offspring fitness
        c. Combine μ parents and λ offspring (μ+λ population)
        d. Select best μ individuals for the next generation (elitist selection)

Complexity:
    - Time: O(T × (n + n² × eval_cost)) where T = max_iterations
    - Space: O(μ × n) for population storage
    - Evaluation: O(n²) for distance matrix lookups

Reference:
    Rechenberg, I. (1973). "Evolutionsstrategie: Optimierung technischer
    Systeme nach Prinzipien der biologischen Evolution."
    Schwefel, H.-P. (1981). "Numerical Optimization of Computer Models."
    John Wiley & Sons.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..other.operators import greedy_insertion
from .params import MuPlusLambdaESParams


class MuPlusLambdaESSolver(PolicyVizMixin):
    """
    (μ+λ) Evolution Strategy solver for VRPP.

    Maintains a population of μ solutions and evolves them via:
    - Recombination: Probabilistically select solution components from archive
    - Mutation: Local perturbation of candidate solutions
    - (μ+λ) Selection: Elitist selection of best μ individuals from combined pool
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MuPlusLambdaESParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Evolution Strategy solver.

        Args:
            dist_matrix: Distance matrix [n+1 × n+1] including depot at index 0.
            wastes: Dictionary mapping node IDs to waste quantities.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            params: ES configuration parameters.
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

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute the (μ+λ) Evolution Strategy.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initialize population (μ individuals)
        population: List[List[List[int]]] = [self._initialize_solution() for _ in range(self.params.population_size)]
        fitness_values = [self._evaluate(solution) for solution in population]

        # Track best solution across all generations
        best_idx = int(np.argmax(fitness_values))
        best_routes = copy.deepcopy(population[best_idx])
        best_profit = fitness_values[best_idx]
        best_cost = self._cost(best_routes)

        # Evolution loop
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Generate λ offspring
            offspring_pool: List[List[List[int]]] = []
            offspring_fitness: List[float] = []

            for _ in range(self.params.offspring_size):
                candidate = self._generate_offspring(population)
                offspring_pool.append(candidate)
                offspring_fitness.append(self._evaluate(candidate))

            # (μ+λ) Selection: Combine parents and offspring
            combined_population = population + offspring_pool
            combined_fitness = fitness_values + offspring_fitness

            # Select best μ individuals
            sorted_indices = np.argsort(combined_fitness)[::-1]  # Sort descending
            survivor_indices = sorted_indices[: self.params.population_size]

            population = [combined_population[i] for i in survivor_indices]
            fitness_values = [combined_fitness[i] for i in survivor_indices]

            # Update global best
            current_best_idx = survivor_indices[0]
            if combined_fitness[current_best_idx] > best_profit:
                best_routes = copy.deepcopy(combined_population[current_best_idx])
                best_profit = combined_fitness[current_best_idx]
                best_cost = self._cost(best_routes)

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                population_size=self.params.population_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Solution Initialization
    # ------------------------------------------------------------------

    def _initialize_solution(self) -> List[List[int]]:
        """
        Initialize a single solution using nearest-neighbor construction.

        Uses randomized node ordering to create diverse initial solutions,
        ensuring the population explores different regions of the search space.

        Returns:
            A feasible routing solution as list of routes.

        Complexity: O(n²) for nearest-neighbor heuristic.
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
    # Offspring Generation (Recombination + Mutation)
    # ------------------------------------------------------------------

    def _generate_offspring(self, population: List[List[List[int]]]) -> List[List[int]]:
        """
        Create offspring via recombination and mutation.

        Recombination Phase:
            With probability p_recomb, select node components from archive solutions.
            Otherwise, select nodes randomly (exploration).

        Mutation Phase:
            With probability p_mut, apply local search perturbation to selected nodes.

        Args:
            population: Current population (archive of μ solutions).

        Returns:
            Offspring solution as list of routes.

        Complexity: O(n) for node selection + O(n²) for greedy insertion.
        """
        candidate_nodes: List[int] = []

        # Flatten population into node pool for recombination
        archive_node_pool: List[List[int]] = []
        for solution in population:
            flat = [n for route in solution for n in route]
            archive_node_pool.append(flat)

        unvisited = set(self.nodes)

        # Build candidate node sequence via recombination
        for _ in self.nodes:
            if not unvisited:
                break

            if self.random.random() < self.params.recombination_rate:
                # Recombination: Select from archive solutions
                source_solution = self.random.choice(archive_node_pool)
                if source_solution:
                    archive_node = self.random.choice(source_solution)
                    selected = archive_node if archive_node in unvisited else self.random.choice(list(unvisited))
                else:
                    selected = self.random.choice(list(unvisited))

                # Mutation: Local neighborhood perturbation
                if self.random.random() < self.params.mutation_rate:
                    neighbors = self._get_nearest_neighbors(selected, unvisited - {selected})
                    if neighbors:
                        selected = neighbors[0]
            else:
                # Random exploration
                selected = self.random.choice(list(unvisited))

            candidate_nodes.append(selected)
            unvisited.discard(selected)

        # Ensure mandatory nodes are included
        for mandatory_node in self.mandatory_nodes:
            if mandatory_node not in candidate_nodes:
                candidate_nodes.append(mandatory_node)

        if not candidate_nodes:
            return []

        # Construct routes via greedy insertion
        try:
            routes = greedy_insertion(
                [],
                candidate_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            # Apply local search refinement
            from logic.src.policies.other.local_search.local_search_aco import ACOLocalSearch

            ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
            routes = ls.optimize(routes)
        except Exception:
            routes = []

        return routes

    def _get_nearest_neighbors(self, node: int, unvisited: set) -> List[int]:
        """
        Return unvisited nodes sorted by Euclidean distance to given node.

        Used for local mutation operator (nearest-neighbor perturbation).

        Args:
            node: Reference node index.
            unvisited: Set of unvisited node indices.

        Returns:
            Sorted list of unvisited nodes (nearest first).

        Complexity: O(n log n) for sorting.
        """
        if not unvisited:
            return []
        return sorted(list(unvisited), key=lambda n: self.dist_matrix[node][n])

    # ------------------------------------------------------------------
    # Fitness Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Compute fitness (net profit) for a routing solution.

        Fitness = Revenue - Cost × C
        Revenue = Σ(waste_collected × R)
        Cost = Total distance traveled

        Args:
            routes: List of routes (each route is a list of node IDs).

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
        Calculate total routing distance (cost).

        Includes depot-to-first-node, inter-node, and last-node-to-depot distances.

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
