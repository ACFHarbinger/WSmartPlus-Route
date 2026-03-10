"""
Hybrid Genetic Search (HGS) policy module.

Combines genetic algorithms with local search and the Split algorithm
for solving the Capacitated Vehicle Routing Problem with Profits.

Reference:
    Vidal, T., Crainic, T. G., Gendreau, M., & Prins, C. (2016). A unified
    solution framework for multi-attribute vehicle routing problems. European
    Journal of Operational Research, 234(3), 658-673.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.operators.crossover import ordered_crossover
from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..other.local_search.local_search_hgs import HGSLocalSearch
from .evolution import evaluate, update_biased_fitness
from .individual import Individual
from .params import HGSParams
from .split import LinearSplit


class HGSSolver(PolicyVizMixin):
    """
    Implements Hybrid Genetic Search for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HGSParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the HGS solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Detailed HGS parameters.
            mandatory_nodes: List of local node indices that MUST be visited.
            seed: Random seed for reproducibility.
        """
        self.d = dist_matrix
        self.wastes = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(seed) if seed is not None else random.Random()

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        self.split_manager = LinearSplit(dist_matrix, wastes, capacity, R, C, params.max_vehicles, mandatory_nodes)
        self.ls = HGSLocalSearch(dist_matrix, wastes, capacity, R, C, params, seed=seed)

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Hybrid Genetic Search algorithm.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        population: List[Individual] = []

        # 1. Initial Population
        for _ in range(self.params.population_size):
            gt = self.nodes[:]
            self.random.shuffle(gt)
            ind = Individual(gt)
            evaluate(ind, self.split_manager)
            population.append(ind)

        current_alpha = self.params.alpha_diversity
        update_biased_fitness(population, self.params.elite_size, current_alpha, self.params.neighbor_list_size)

        start_time = time.process_time()
        it = 0
        last_improvement_it = 0
        best_profit_so_far = max(ind.profit_score for ind in population)
        while it < self.params.n_generations:
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break
            it += 1
            # 2. Selection & Crossover
            p1, p2 = self._select_parents(population)
            child = ordered_crossover(p1, p2, rng=self.random)

            # 3. Local Search (Mutation)
            if self.random.random() < self.params.mutation_rate:
                evaluate(child, self.split_manager)
                child = self.ls.optimize(child)

            evaluate(child, self.split_manager)
            population.append(child)

            if child.profit_score > best_profit_so_far:
                best_profit_so_far = child.profit_score
                last_improvement_it = it

            # Adaptive alpha diversity
            # Calculate current population diversity
            avg_dist = np.mean([ind.dist_to_parents for ind in population])
            if avg_dist < self.params.min_diversity_threshold:
                current_alpha = min(1.0, current_alpha + self.params.diversity_change_rate)
            elif it - last_improvement_it > self.params.no_improvement_threshold:
                current_alpha = max(0.0, current_alpha - self.params.diversity_change_rate)

            # 4. Survivor Selection
            if len(population) > self.params.population_size * self.params.survivor_threshold:
                update_biased_fitness(population, self.params.elite_size, current_alpha, self.params.neighbor_list_size)
                population.sort(key=lambda x: x.fitness)
                population = population[: self.params.population_size]

            self._viz_record(
                iteration=it,
                best_profit=best_profit_so_far,
                child_profit=child.profit_score,
                child_cost=child.cost,
                population_size=len(population),
            )

        update_biased_fitness(population, self.params.elite_size, current_alpha, self.params.neighbor_list_size)
        best_ind = min(population, key=lambda x: -x.profit_score)

        return best_ind.routes, best_ind.profit_score, best_ind.cost

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        # Binary Tournament
        def tournament():
            """Perform a binary tournament selection."""
            i1, i2 = self.random.sample(population, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()
