"""
Pure Island Model Genetic Algorithm for VRPP.

Multi-population GA with migration but WITHOUT local search (not memetic).
Replaces "Soccer League Competition (SLC)":
- "Soccer Teams" → Sub-populations (islands)
- "Players" → Chromosomes
- "Matches" → Fitness evaluations
- "Player Transfers" → Migration operators
- "Relegation" → Replacement via tournament selection

Algorithm:
    1. Initialize K islands with N chromosomes each
    2. For each generation:
        a. **Selection**: Tournament selection within each island
        b. **Crossover**: Order-preserving recombination
        c. **Mutation**: Destroy-repair perturbation
        d. **Replacement**: Elitist strategy (keep best)
        e. **Migration**: Periodic exchange of best solutions (ring topology)

Complexity:
    - Time: O(T × K × N × n²) where T = generations
    - Space: O(K × N × n) for island populations
    - Migration: O(K × migration_size) per interval

Reference:
    Moosavian, N., & Rppdsarou, B. K. (2014).
    "Soccer league competition algorithm: A novel meta-heuristic
    algorithm for optimal design of water distribution networks."
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..other.operators import greedy_insertion, random_removal
from .params import PureIslandModelGAParams


class PureIslandModelGASolver(PolicyVizMixin):
    """
    Pure Island Model GA solver for VRPP (without local search).

    Maintains K independent sub-populations that evolve via genetic
    operators only, with periodic migration between islands.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: PureIslandModelGAParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Pure Island Model GA solver.

        Args:
            dist_matrix: Distance matrix [n+1 × n+1] including depot at index 0.
            wastes: Dictionary mapping node IDs to waste quantities.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            params: Island Model GA configuration parameters.
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
        Execute Pure Island Model Genetic Algorithm.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # Initialize islands (K sub-populations)
        islands: List[List[Tuple[List[List[int]], float]]] = []
        for _ in range(self.params.n_islands):
            island_population = []
            for _ in range(self.params.island_size):
                chromosome = self._initialize_chromosome()
                fitness = self._evaluate_fitness(chromosome)
                island_population.append((chromosome, fitness))
            islands.append(island_population)

        # Track global best across all islands
        best_chromosome, best_fitness = self._get_global_best(islands)
        best_cost = self._calculate_cost(best_chromosome)

        # Evolution loop
        for generation in range(self.params.max_generations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Evolve each island independently
            for island_idx in range(self.params.n_islands):
                islands[island_idx] = self._evolve_island(islands[island_idx])

            # Update global best
            gen_best_chromosome, gen_best_fitness = self._get_global_best(islands)
            if gen_best_fitness > best_fitness:
                best_chromosome = copy.deepcopy(gen_best_chromosome)
                best_fitness = gen_best_fitness
                best_cost = self._calculate_cost(best_chromosome)

            # Migration (periodic)
            if generation > 0 and generation % self.params.migration_interval == 0:
                islands = self._migrate(islands)

            self._viz_record(
                iteration=generation,
                best_profit=best_fitness,
                best_cost=best_cost,
                n_islands=self.params.n_islands,
            )

        return best_chromosome, best_fitness, best_cost

    # ------------------------------------------------------------------
    # Island Evolution (Pure Genetic Operators)
    # ------------------------------------------------------------------

    def _evolve_island(
        self, island_population: List[Tuple[List[List[int]], float]]
    ) -> List[Tuple[List[List[int]], float]]:
        """
        Evolve a single island for one generation using genetic operators only.

        Phases:
            1. Tournament Selection
            2. Crossover
            3. Mutation
            4. Elitist Replacement

        Args:
            island_population: Current island population.

        Returns:
            Updated island population.

        Complexity: O(N × n²) where N = island_size.
        """
        # Phase 1: Tournament Selection
        mating_pool = self._tournament_selection(island_population)

        # Phase 2: Crossover
        offspring = []
        for i in range(0, len(mating_pool) - 1, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i + 1]

            if self.rng.random() < self.params.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            offspring.extend([child1, child2])

        # Handle odd mating pool size
        if len(mating_pool) % 2 == 1:
            offspring.append(copy.deepcopy(mating_pool[-1]))

        # Phase 3: Mutation
        for i in range(len(offspring)):
            if self.rng.random() < self.params.mutation_rate:
                offspring[i] = self._mutate(offspring[i])

        # Evaluate offspring fitness
        offspring_with_fitness = [(child, self._evaluate_fitness(child)) for child in offspring]

        # Phase 4: Elitist Replacement
        new_population = self._elitist_replacement(island_population, offspring_with_fitness)

        return new_population

    # ------------------------------------------------------------------
    # Genetic Operators
    # ------------------------------------------------------------------

    def _tournament_selection(self, population: List[Tuple[List[List[int]], float]]) -> List[List[List[int]]]:
        """
        Standard tournament selection.

        Args:
            population: Current population with fitness values.

        Returns:
            Mating pool (selected parents).

        Complexity: O(N × k) where k = tournament_size.
        """
        mating_pool = []
        for _ in range(len(population)):
            tournament = self.rng.sample(population, min(self.params.tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x[1])
            mating_pool.append(copy.deepcopy(winner[0]))
        return mating_pool

    def _crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Order-preserving crossover for routing solutions.

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            Tuple of (child1, child2).

        Complexity: O(n²) for route construction.
        """
        nodes_p1 = {n for route in parent1 for n in route}
        nodes_p2 = {n for route in parent2 for n in route}

        # Child 1: Inherit from p1, supplement with p2
        child1_nodes = list(nodes_p1)
        for node in nodes_p2:
            if node not in nodes_p1:
                child1_nodes.append(node)

        # Child 2: Inherit from p2, supplement with p1
        child2_nodes = list(nodes_p2)
        for node in nodes_p1:
            if node not in nodes_p2:
                child2_nodes.append(node)

        child1 = self._construct_routes(child1_nodes)
        child2 = self._construct_routes(child2_nodes)

        return child1, child2

    def _mutate(self, chromosome: List[List[int]]) -> List[List[int]]:
        """
        Destroy-repair mutation operator.

        Args:
            chromosome: Chromosome to mutate.

        Returns:
            Mutated chromosome.

        Complexity: O(n²).
        """
        try:
            n_remove = max(3, int(0.15 * self.n_nodes))
            partial, removed = random_removal(chromosome, n_remove, rng=self.rng)
            mutated = greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            return mutated
        except Exception:
            return copy.deepcopy(chromosome)

    def _elitist_replacement(
        self,
        parents: List[Tuple[List[List[int]], float]],
        offspring: List[Tuple[List[List[int]], float]],
    ) -> List[Tuple[List[List[int]], float]]:
        """
        Elitist replacement: Keep top N individuals from parents + offspring.

        Args:
            parents: Parent population with fitness.
            offspring: Offspring population with fitness.

        Returns:
            New population.

        Complexity: O(N log N) for sorting.
        """
        combined = parents + offspring
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[: self.params.island_size]

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _migrate(self, islands: List[List[Tuple[List[List[int]], float]]]) -> List[List[Tuple[List[List[int]], float]]]:
        """
        Perform inter-island migration (ring topology).

        Each island sends its best solutions to the next island.

        Args:
            islands: All island populations.

        Returns:
            Updated islands after migration.

        Complexity: O(K × migration_size).
        """
        migrants = []
        for island in islands:
            # Sort island by fitness and extract best individuals
            sorted_island = sorted(island, key=lambda x: x[1], reverse=True)
            best_individuals = sorted_island[: self.params.migration_size]
            migrants.append(best_individuals)

        # Ring topology: Island i receives from island (i-1) mod K
        for i in range(self.params.n_islands):
            source_idx = (i - 1) % self.params.n_islands

            # Replace worst individuals in target island with migrants
            islands[i].sort(key=lambda x: x[1], reverse=True)
            for j, migrant in enumerate(migrants[source_idx]):
                replacement_idx = -(j + 1)  # Replace from worst
                if abs(replacement_idx) <= len(islands[i]):
                    islands[i][replacement_idx] = copy.deepcopy(migrant)

        return islands

    # ------------------------------------------------------------------
    # Utility Functions
    # ------------------------------------------------------------------

    def _initialize_chromosome(self) -> List[List[int]]:
        """
        Initialize a single chromosome using nearest-neighbor heuristic.

        Returns:
            A feasible routing solution.

        Complexity: O(n²).
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
            rng=self.rng,
        )
        return routes

    def _construct_routes(self, nodes: List[int]) -> List[List[int]]:
        """
        Construct routes from node list via greedy insertion.

        Args:
            nodes: Nodes to visit.

        Returns:
            Routing solution.

        Complexity: O(n²).
        """
        try:
            routes = greedy_insertion(
                [],
                nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            return routes
        except Exception:
            return []

    def _get_global_best(self, islands: List[List[Tuple[List[List[int]], float]]]) -> Tuple[List[List[int]], float]:
        """
        Find global best solution across all islands.

        Args:
            islands: All island populations.

        Returns:
            Tuple of (best_chromosome, best_fitness).

        Complexity: O(K × N).
        """
        global_best = None
        global_best_fitness = float("-inf")

        for island in islands:
            island_best = max(island, key=lambda x: x[1])
            if island_best[1] > global_best_fitness:
                global_best = island_best[0]
                global_best_fitness = island_best[1]

        return copy.deepcopy(global_best), global_best_fitness

    def _evaluate_fitness(self, routes: List[List[int]]) -> float:
        """
        Evaluate chromosome fitness (net profit).

        Args:
            routes: Routing solution.

        Returns:
            Net profit.

        Complexity: O(n).
        """
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(n, 0.0) * self.R for route in routes for n in route)
        return revenue - self._calculate_cost(routes) * self.C

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing distance.

        Args:
            routes: List of routes.

        Returns:
            Total distance.

        Complexity: O(n).
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
