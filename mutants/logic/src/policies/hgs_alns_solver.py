"""
HGS-ALNS Hybrid Solver.

This module implements a hybrid approach where Hybrid Genetic Search (HGS)
uses Adaptive Large Neighborhood Search (ALNS) for its education phase.
"""

import random
import time
from typing import Dict, List, Tuple

import numpy as np

from .adaptive_large_neighborhood_search.alns import ALNSSolver
from .adaptive_large_neighborhood_search.params import ALNSParams
from .hybrid_genetic_search.evolution import evaluate, ordered_crossover, update_biased_fitness
from .hybrid_genetic_search.hgs import HGSSolver
from .hybrid_genetic_search.types import HGSParams, Individual


class HGSALNSSolver(HGSSolver):
    """
    Hybrid solver that combines HGS with ALNS as a local search optimizer.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        demands: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HGSParams,
        alns_education_iterations: int = 50,
    ):
        """
        Initialize the hybrid HGS-ALNS solver.

        Args:
            dist_matrix: NxN distance matrix.
            demands: Dictionary of node demands.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Detailed HGS parameters.
            alns_education_iterations: Number of ALNS iterations used during education.
        """
        super().__init__(dist_matrix, demands, capacity, R, C, params)
        self.alns_iter = alns_education_iterations

        # Initialize ALNS solver with limited iterations for intensive education
        alns_params = ALNSParams(
            max_iterations=self.alns_iter,
            time_limit=max(1, params.time_limit // 10),  # Heuristic limit
        )
        self.alns_solver = ALNSSolver(dist_matrix, demands, capacity, R, C, alns_params)

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Hybrid Genetic Search algorithm with ALNS-based education.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        population: List[Individual] = []

        # 1. Initial Population
        for _ in range(self.params.population_size):
            gt = self.nodes[:]
            random.shuffle(gt)
            ind = Individual(gt)
            evaluate(ind, self.split_manager)
            population.append(ind)

        update_biased_fitness(population, self.params.elite_size)

        start_time = time.time()
        it = 0
        while time.time() - start_time < self.params.time_limit:
            it += 1
            # 2. Selection & Crossover
            p1, p2 = self._select_parents(population)
            child = ordered_crossover(p1, p2)

            # 3. Hybrid Education (Mutation with ALNS)
            if random.random() < self.params.mutation_rate:
                # Need to have routes assigned before ALNS
                evaluate(child, self.split_manager)
                if child.routes:
                    # Convert child to ALNS format and run ALNS
                    # ALNSSolver.solve accepts List[List[int]] as initial_solution
                    new_routes, profit, cost = self.alns_solver.solve(initial_solution=child.routes)

                    # Update individual with ALNS results
                    child.routes = new_routes
                    child.profit_score = profit
                    child.cost = cost

                    # Reconstruct giant tour from ALNS routes to maintain HGS consistency
                    new_gt = []
                    for r in new_routes:
                        new_gt.extend(r)
                    child.giant_tour = new_gt
                else:
                    # Fallback to standard local search if split failed to produce routes
                    child = self.ls.optimize(child)

            evaluate(child, self.split_manager)
            population.append(child)

            # 4. Survivor Selection
            if len(population) > self.params.population_size * 2:
                update_biased_fitness(population, self.params.elite_size)
                population.sort(key=lambda x: x.fitness)
                population = population[: self.params.population_size]

        update_biased_fitness(population, self.params.elite_size)
        best_ind = min(population, key=lambda x: -x.profit_score)

        return best_ind.routes, best_ind.profit_score, best_ind.cost
