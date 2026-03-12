"""
(μ,λ) Evolution Strategy for VRPP.

Multi-phase evolution strategy with local search, probabilistic selection, and
random restarts. Replaces the metaphor-based "Artificial Bee Colony":
- "Employed bees" → Local search agents (localized exploitation)
- "Onlooker bees" → Fitness-proportional offspring selection
- "Scout bees" → Random restart mechanism for stagnant solutions
- "Food sources" → Parent solutions in population

Algorithm:
    1. Initialize population of μ solutions
    2. For each iteration:
        a. **Local Search Phase**: Each parent generates offspring via mutation
        b. **Selection Phase**: Select offspring via fitness-proportional selection
        c. **Update Phase**: Replace parents with selected offspring (non-elitist)
        d. **Restart Phase**: Replace stagnant solutions with random restarts

Complexity:
    - Time: O(T × λ × n²) where T = iterations, λ = offspring count
    - Space: O(μ × n) for population + O(λ × n) for offspring generation
    - Local search: O(n²) per offspring for greedy insertion

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

from ..other.operators import greedy_insertion, random_removal
from .params import MuCommaLambdaESParams


class MuCommaLambdaESSolver(PolicyVizMixin):
    """
    (μ,λ) Evolution Strategy solver for VRPP.

    Maintains μ parent solutions and generates λ offspring per iteration.
    Selection is non-elitist: parents are replaced entirely by offspring.
    Includes random restart mechanism for stagnant solutions.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MuCommaLambdaESParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the (μ,λ) Evolution Strategy solver.

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
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute (μ,λ) Evolution Strategy.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initialize population of μ parents
        parents = [self._initialize_solution() for _ in range(self.params.population_size)]
        fitness_values = [self._evaluate(solution) for solution in parents]
        stagnation_counters = [0] * self.params.population_size

        # Track global best across all generations
        best_idx = int(np.argmax(fitness_values))
        best_routes = copy.deepcopy(parents[best_idx])
        best_profit = fitness_values[best_idx]
        best_cost = self._cost(best_routes)

        # Evolution loop
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # --- Phase 1: Local Search (Exploitation) ---
            # Each parent generates offspring via localized mutation
            for i in range(self.params.population_size):
                # Select a peer parent for crossover guidance
                peer_idx = self.rng.choice([x for x in range(self.params.population_size) if x != i])
                offspring = self._generate_offspring(parents[i], parents[peer_idx])
                offspring_fitness = self._evaluate(offspring)

                # Greedy offspring acceptance
                if offspring_fitness > fitness_values[i]:
                    parents[i] = offspring
                    fitness_values[i] = offspring_fitness
                    stagnation_counters[i] = 0
                else:
                    stagnation_counters[i] += 1

            # --- Phase 2: Probabilistic Selection (Onlooker Phase) ---
            # Generate λ additional offspring via fitness-proportional selection
            selection_probabilities = self._compute_selection_probabilities(fitness_values)

            for _ in range(self.params.population_size):
                i = self._roulette_wheel_selection(selection_probabilities)
                peer_idx = self.rng.choice([x for x in range(self.params.population_size) if x != i])
                offspring = self._generate_offspring(parents[i], parents[peer_idx])
                offspring_fitness = self._evaluate(offspring)

                # Greedy offspring acceptance
                if offspring_fitness > fitness_values[i]:
                    parents[i] = offspring
                    fitness_values[i] = offspring_fitness
                    stagnation_counters[i] = 0
                else:
                    stagnation_counters[i] += 1

            # Update global best
            for i in range(self.params.population_size):
                if fitness_values[i] > best_profit:
                    best_routes = copy.deepcopy(parents[i])
                    best_profit = fitness_values[i]
                    best_cost = self._cost(best_routes)

            # --- Phase 3: Random Restart (Scout Phase) ---
            # Replace stagnant solutions with random restarts
            for i in range(self.params.population_size):
                if stagnation_counters[i] > self.params.stagnation_limit:
                    parents[i] = self._initialize_solution()
                    fitness_values[i] = self._evaluate(parents[i])
                    stagnation_counters[i] = 0

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
        Initialize a single solution using nearest-neighbor heuristic.

        Randomized node ordering creates diverse initial population.

        Returns:
            A feasible routing solution as list of routes.

        Complexity: O(n²) for nearest-neighbor construction.
        """
        from logic.src.policies.other.operators.heuristics.initialization import build_nn_routes

        routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.rng,
        )
        return routes

    # ------------------------------------------------------------------
    # Offspring Generation (Mutation + Crossover)
    # ------------------------------------------------------------------

    def _generate_offspring(self, parent: List[List[int]], peer: List[List[int]]) -> List[List[int]]:
        """
        Generate offspring via cross-solution recombination and mutation.

        Recombination:
            Extract nodes from peer solution and inject into parent.
            Mimics the continuous formula: v_ij = x_ij + φ(x_ij - x_kj)

        Mutation:
            Additional random removal and greedy reinsertion for diversity.

        Args:
            parent: Current parent solution.
            peer: Peer solution for crossover guidance.

        Returns:
            Offspring solution.

        Complexity: O(n²) for destroy-repair operation.
        """
        if not parent or not peer:
            return copy.deepcopy(parent)

        n_remove = max(3, self.params.n_removal)

        # Extract nodes from peer for recombination
        peer_nodes = [node for route in peer for node in route]
        if not peer_nodes:
            return copy.deepcopy(parent)

        selected_peer_nodes = self.rng.sample(peer_nodes, min(n_remove, len(peer_nodes)))

        # Remove peer nodes from parent
        offspring = copy.deepcopy(parent)
        for route in offspring:
            for node in selected_peer_nodes:
                if node in route:
                    route.remove(node)

        offspring = [route for route in offspring if route]

        # Apply additional mutation (random removal)
        try:
            partial_routes, additional_removed = random_removal(offspring, n_remove, rng=self.rng)
            nodes_to_insert = sorted(list(set(selected_peer_nodes + additional_removed)))

            repaired_routes = greedy_insertion(
                partial_routes,
                nodes_to_insert,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )

            # Apply local search refinement
            from logic.src.policies.other.local_search.local_search_aco import ACOLocalSearch

            ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
            return ls.optimize(repaired_routes)
        except Exception:
            return copy.deepcopy(parent)

    # ------------------------------------------------------------------
    # Selection Mechanisms
    # ------------------------------------------------------------------

    def _compute_selection_probabilities(self, fitness_values: List[float]) -> List[float]:
        """
        Compute fitness-proportional selection probabilities (roulette wheel).

        Shifts fitness values to ensure all probabilities are positive.

        Args:
            fitness_values: List of fitness values for all parents.

        Returns:
            List of selection probabilities summing to 1.0.

        Complexity: O(μ) for normalization.
        """
        min_fitness = min(fitness_values)
        shifted_fitness = [f - min_fitness + 1e-9 for f in fitness_values]
        total_fitness = sum(shifted_fitness)
        return [f / total_fitness for f in shifted_fitness]

    def _roulette_wheel_selection(self, probabilities: List[float]) -> int:
        """
        Select an index via roulette wheel selection.

        Args:
            probabilities: List of selection probabilities.

        Returns:
            Selected index.

        Complexity: O(μ) for cumulative probability traversal.
        """
        r = self.rng.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return i
        return len(probabilities) - 1

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
