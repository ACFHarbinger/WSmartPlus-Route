"""
Genetic Algorithm with Stochastic Tournament Selection for VRPP.

Canonical GA where selection is performed via pairwise stochastic tournaments.
Replaces the metaphor-based "League Championship Algorithm (LCA)":
- "League Schedule" → Pairwise fitness evaluation cycles
- "Playing Strength" → Fitness score
- "Match Outcome" → Stochastic tournament (sigmoid probability)
- "Team Formation" → Crossover operator

Algorithm:
    1. Initialize population of N chromosomes (routing solutions)
    2. For each generation:
        a. **Evaluation**: Compute fitness for all individuals
        b. **Selection**: Stochastic tournament selection
            - Each individual competes against k random opponents
            - Win probability: P(i > j) = σ(β × (f_i - f_j))
            - Select winners for mating pool
        c. **Crossover**: Apply order-preserving recombination
        d. **Mutation**: Apply destroy-repair perturbation
        e. **Replacement**: Elitist strategy (keep top individuals)

Complexity:
    - Time: O(T × N × k × eval_cost) where T = generations, k = tournament_size
    - Space: O(N × n) for population storage
    - Selection: O(N × k) for tournament comparisons

Reference:
    Kashan, A. H. (2013). "League Championship Algorithm (LCA): An
    algorithm for global optimization inspired by sport championships."
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..other.operators import greedy_insertion, random_removal
from .params import StochasticTournamentGAParams


class StochasticTournamentGASolver(PolicyVizMixin):
    """
    Genetic Algorithm with Stochastic Tournament Selection for VRPP.

    Uses pairwise tournaments with sigmoid win probability for selection pressure.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: StochasticTournamentGAParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Stochastic Tournament GA solver.

        Args:
            dist_matrix: Distance matrix [n+1 × n+1] including depot at index 0.
            wastes: Dictionary mapping node IDs to waste quantities.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            params: GA configuration parameters.
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
        Execute Stochastic Tournament Genetic Algorithm.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # Phase 1: Initialization
        population = [self._initialize_chromosome() for _ in range(self.params.population_size)]
        fitness_values = [self._evaluate_fitness(chromosome) for chromosome in population]

        # Track global best
        best_idx = int(np.argmax(fitness_values))
        best_chromosome = copy.deepcopy(population[best_idx])
        best_fitness = fitness_values[best_idx]
        best_cost = self._calculate_cost(best_chromosome)

        # Phase 2: Evolution Loop
        for generation in range(self.params.max_generations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Phase 2a: Stochastic Tournament Selection
            mating_pool = self._tournament_selection(population, fitness_values)

            # Phase 2b: Crossover (Recombination)
            offspring_population = []
            for i in range(0, len(mating_pool) - 1, 2):
                parent1 = mating_pool[i]
                parent2 = mating_pool[i + 1]

                if self.rng.random() < self.params.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

                offspring_population.extend([child1, child2])

            # Handle odd population size
            if len(mating_pool) % 2 == 1:
                offspring_population.append(copy.deepcopy(mating_pool[-1]))

            # Phase 2c: Mutation
            for i in range(len(offspring_population)):
                if self.rng.random() < self.params.mutation_rate:
                    offspring_population[i] = self._mutate(offspring_population[i])

            # Phase 2d: Fitness Evaluation
            offspring_fitness = [self._evaluate_fitness(child) for child in offspring_population]

            # Phase 2e: Elitist Replacement
            population, fitness_values = self._elitist_replacement(
                population, fitness_values, offspring_population, offspring_fitness
            )

            # Update global best
            gen_best_idx = int(np.argmax(fitness_values))
            if fitness_values[gen_best_idx] > best_fitness:
                best_chromosome = copy.deepcopy(population[gen_best_idx])
                best_fitness = fitness_values[gen_best_idx]
                best_cost = self._calculate_cost(best_chromosome)

            self._viz_record(
                iteration=generation,
                best_profit=best_fitness,
                best_cost=best_cost,
                population_size=self.params.population_size,
            )

        return best_chromosome, best_fitness, best_cost

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_chromosome(self) -> List[List[int]]:
        """
        Initialize a single chromosome (routing solution).

        Uses nearest-neighbor heuristic with randomized node ordering
        to create diverse initial population.

        Returns:
            A feasible routing solution as list of routes.

        Complexity: O(n²) for construction.
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
    # Selection (Stochastic Tournament)
    # ------------------------------------------------------------------

    def _tournament_selection(
        self, population: List[List[List[int]]], fitness_values: List[float]
    ) -> List[List[List[int]]]:
        """
        Perform stochastic tournament selection.

        Each individual competes against k random opponents. Win probability
        is determined by sigmoid function of fitness difference:
            P(i defeats j) = σ(β × (fitness_i - fitness_j))
        where σ(x) = 1/(1 + exp(-x))

        Individuals with more wins are more likely to be selected for mating pool.

        Args:
            population: Current population of chromosomes.
            fitness_values: Fitness scores for all chromosomes.

        Returns:
            Mating pool (selected parents for reproduction).

        Complexity: O(N × k) where k = tournament_competitors.
        """
        tournament_scores = [0.0] * self.params.population_size

        # Each individual competes against k random opponents
        for i in range(self.params.population_size):
            opponents = self.rng.sample(
                range(self.params.population_size),
                min(self.params.tournament_competitors, self.params.population_size - 1),
            )

            for j in opponents:
                if i == j:
                    continue

                # Stochastic tournament: Win probability via sigmoid
                fitness_diff = fitness_values[i] - fitness_values[j]
                win_probability = self._sigmoid(self.params.selection_pressure * fitness_diff)

                if self.rng.random() < win_probability:
                    tournament_scores[i] += 1

        # Select top N/2 individuals based on tournament scores (with ties broken by fitness)
        scored_individuals = list(zip(tournament_scores, fitness_values, range(self.params.population_size)))
        scored_individuals.sort(reverse=True)

        mating_pool_size = self.params.population_size
        mating_pool = [copy.deepcopy(population[idx]) for _, _, idx in scored_individuals[:mating_pool_size]]

        return mating_pool

    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Compute sigmoid activation: σ(x) = 1 / (1 + exp(-x)).

        Args:
            x: Input value.

        Returns:
            Sigmoid output ∈ [0, 1].

        Complexity: O(1).
        """
        return 1.0 / (1.0 + math.exp(-x))

    # ------------------------------------------------------------------
    # Crossover (Recombination)
    # ------------------------------------------------------------------

    def _crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Apply order-preserving crossover for routing solutions.

        Extracts node subsets from each parent and recombines them via
        greedy insertion to maintain feasibility.

        Args:
            parent1: First parent chromosome.
            parent2: Second parent chromosome.

        Returns:
            Tuple of (child1, child2) offspring chromosomes.

        Complexity: O(n²) for greedy insertion.
        """
        # Extract node sets from parents
        nodes_p1 = {n for route in parent1 for n in route}
        nodes_p2 = {n for route in parent2 for n in route}

        # Child 1: Inherit from parent1, supplement with parent2
        child1_nodes = list(nodes_p1)
        for node in nodes_p2:
            if node not in nodes_p1 and len(child1_nodes) < self.n_nodes:
                child1_nodes.append(node)

        # Child 2: Inherit from parent2, supplement with parent1
        child2_nodes = list(nodes_p2)
        for node in nodes_p1:
            if node not in nodes_p2 and len(child2_nodes) < self.n_nodes:
                child2_nodes.append(node)

        # Construct routes via greedy insertion
        child1 = self._construct_routes(child1_nodes)
        child2 = self._construct_routes(child2_nodes)

        return child1, child2

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, chromosome: List[List[int]]) -> List[List[int]]:
        """
        Apply mutation operator (destroy-repair).

        Removes random nodes and reinserts them greedily to maintain
        feasibility while introducing variation.

        Args:
            chromosome: Chromosome to mutate.

        Returns:
            Mutated chromosome.

        Complexity: O(n²) for destroy-repair.
        """
        try:
            n_remove = max(3, int(0.1 * self.n_nodes))
            partial_routes, removed_nodes = random_removal(chromosome, n_remove, rng=self.rng)
            mutated_routes = greedy_insertion(
                partial_routes,
                removed_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            return mutated_routes
        except Exception:
            return copy.deepcopy(chromosome)

    # ------------------------------------------------------------------
    # Replacement (Elitist)
    # ------------------------------------------------------------------

    def _elitist_replacement(
        self,
        parents: List[List[List[int]]],
        parent_fitness: List[float],
        offspring: List[List[List[int]]],
        offspring_fitness: List[float],
    ) -> Tuple[List[List[List[int]]], List[float]]:
        """
        Elitist replacement strategy.

        Combine parents and offspring, then select top N individuals.
        Preserves best solutions across generations.

        Args:
            parents: Parent population.
            parent_fitness: Parent fitness values.
            offspring: Offspring population.
            offspring_fitness: Offspring fitness values.

        Returns:
            Tuple of (new_population, new_fitness_values).

        Complexity: O(N log N) for sorting.
        """
        # Combine parents and offspring
        combined_population = parents + offspring
        combined_fitness = parent_fitness + offspring_fitness

        # Sort by fitness (descending)
        sorted_indices = np.argsort(combined_fitness)[::-1]

        # Select top N individuals
        new_population = [combined_population[i] for i in sorted_indices[: self.params.population_size]]
        new_fitness = [combined_fitness[i] for i in sorted_indices[: self.params.population_size]]

        return new_population, new_fitness

    # ------------------------------------------------------------------
    # Utility Functions
    # ------------------------------------------------------------------

    def _construct_routes(self, nodes: List[int]) -> List[List[int]]:
        """
        Construct routes from node list via greedy insertion.

        Args:
            nodes: List of nodes to visit.

        Returns:
            Routing solution as list of routes.

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

    def _evaluate_fitness(self, routes: List[List[int]]) -> float:
        """
        Evaluate chromosome fitness (net profit).

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
        return revenue - self._calculate_cost(routes) * self.C

    def _calculate_cost(self, routes: List[List[int]]) -> float:
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
