"""
Island Model Genetic Algorithm with Stochastic Tournaments (IMGA-ST) for VRPP.

A mathematically rigorous replacement for metaphor-based sports algorithms (VPL, SLC).
This solver implements a multi-island genetic algorithm where sub-populations evolve
independently and periodically migrate solutions. Selection is performed via
stochastic tournaments with sigmoid win probabilities.

Key Features:
    - Island Model: Distributed population management (K islands of N individuals).
    - Stochastic Tournament Selection: Pairwise competition using sigmoid probability.
    - Ring Migration: Periodic exchange of elite solutions between adjacent islands.
    - Local Improvement: ALNS intensification applied to offspring.
    - Non-metaphorical Terminology: Standard OR/EC concepts (population, selection, migration).

Reference:
    Whitley, D., et al. (1998). "The island model genetic algorithm."
    Goldberg, D. E., & Deb, K. (1991). "A comparative analysis of selection schemes."
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..hybrid_genetic_search.individual import Individual
from ..other.operators import build_nn_routes, greedy_profit_insertion, ordered_crossover
from .params import IslandModelSTGAParams


class IslandModelSTGASolver(PolicyVizMixin):
    """
    Island Model Genetic Algorithm with Stochastic Tournaments (IMGA-ST).
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: IslandModelSTGAParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
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
        Execute the IMGA-ST optimization.
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # Initialize islands
        islands: List[List[List[List[int]]]] = []
        island_fitness: List[List[float]] = []

        for _ in range(self.params.n_islands):
            pop = [self._initialize_solution() for _ in range(self.params.island_size)]
            islands.append(pop)
            island_fitness.append([self._evaluate(sol) for sol in pop])

        # Track global best
        flat_pops = [sol for pop in islands for sol in pop]
        flat_fits = [fit for pop_fit in island_fitness for fit in pop_fit]
        best_idx = int(np.argmax(flat_fits))
        best_routes = copy.deepcopy(flat_pops[best_idx])
        best_profit = flat_fits[best_idx]
        best_cost = self._cost(best_routes)

        for gen in range(self.params.max_generations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Migration Phase (Ring Topology)
            if gen > 0 and gen % self.params.migration_interval == 0:
                self._migrate(islands, island_fitness)

            # Evolution Phase
            for k in range(self.params.n_islands):
                islands[k], island_fitness[k] = self._evolve_island(islands[k], island_fitness[k])

            # Update Global Best
            for k in range(self.params.n_islands):
                max_fit_idx = int(np.argmax(island_fitness[k]))
                if island_fitness[k][max_fit_idx] > best_profit:
                    best_routes = copy.deepcopy(islands[k][max_fit_idx])
                    best_profit = island_fitness[k][max_fit_idx]
                    best_cost = self._cost(best_routes)

            self._viz_record(
                iteration=gen,
                best_profit=best_profit,
                best_cost=best_cost,
                n_islands=self.params.n_islands,
                island_size=self.params.island_size,
            )

        return best_routes, best_profit, best_cost

    def _initialize_solution(self) -> List[List[int]]:
        """Initialize solution using NN heuristic."""
        return build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.random,
        )

    def _migrate(self, islands: List[List[List[List[int]]]], island_fitness: List[List[float]]):
        """Perform ring migration between islands."""
        migrants = []
        migrant_fitness = []

        # Collect best individuals from each island
        for k in range(self.params.n_islands):
            sorted_indices = np.argsort(island_fitness[k])[::-1]
            best_idx = sorted_indices[: self.params.migration_size]
            migrants.append([copy.deepcopy(islands[k][i]) for i in best_idx])
            migrant_fitness.append([island_fitness[k][i] for i in best_idx])

        # Move migrants to next island (i+1) mod K, replacing worst
        for k in range(self.params.n_islands):
            dest_k = (k + 1) % self.params.n_islands
            sorted_indices = np.argsort(island_fitness[dest_k])  # Worst first

            for i in range(self.params.migration_size):
                replace_idx = sorted_indices[i]
                islands[dest_k][replace_idx] = migrants[k][i]
                island_fitness[dest_k][replace_idx] = migrant_fitness[k][i]

    def _evolve_island(
        self, population: List[List[List[int]]], fitness: List[float]
    ) -> Tuple[List[List[List[int]]], List[float]]:
        """Evolution step for a single island."""
        new_population = []
        new_fitness = []

        # Elitism: Preserve best
        sorted_indices = np.argsort(fitness)[::-1]
        for i in range(self.params.elitism_size):
            idx = sorted_indices[i]
            new_population.append(copy.deepcopy(population[idx]))
            new_fitness.append(fitness[idx])

        # Generate offspring
        while len(new_population) < self.params.island_size:
            # Selection
            parent1 = self._stochastic_tournament(population, fitness)
            parent2 = self._stochastic_tournament(population, fitness)

            # Recombination
            if self.random.random() < self.params.crossover_rate:
                offspring_nodes = self._crossover(parent1, parent2)
            else:
                offspring_nodes = [n for r in parent1 for n in r]

            # Mutation
            if self.random.random() < self.params.mutation_rate:
                offspring_nodes = self._mutate(offspring_nodes)

            # Reconstruction and Local Improvement
            offspring = self._improve(offspring_nodes)
            new_population.append(offspring)
            new_fitness.append(self._evaluate(offspring))

        return new_population, new_fitness

    def _stochastic_tournament(self, population: List[List[List[int]]], fitness: List[float]) -> List[List[int]]:
        """Pairwise tournament with sigmoid win probability."""
        candidates = self.random.sample(range(len(population)), self.params.tournament_size)
        best_idx = candidates[0]

        for i in range(1, len(candidates)):
            curr_idx = candidates[i]
            # Win probability: P(curr wins against best) = 1 / (1 + exp(-beta * (f_curr - f_best)))
            # Higher fitness is better
            diff = fitness[curr_idx] - fitness[best_idx]
            # Clip diff to prevent overflow in exp
            win_prob = 1.0 / (1.0 + np.exp(-self.params.selection_pressure * np.clip(diff, -100, 100)))

            if self.random.random() < win_prob:
                best_idx = curr_idx

        return population[best_idx]

    def _crossover(self, p1: List[List[int]], p2: List[List[int]]) -> List[int]:
        """Ordered crossover on node sequences."""
        nodes1 = [n for r in p1 for n in r]
        nodes2 = [n for r in p2 for n in r]

        if not nodes1 or not nodes2:
            return nodes1 or nodes2

        # Wrap in Individual for ordered_crossover compatibility
        ind1 = Individual(nodes1)
        ind2 = Individual(nodes2)

        child_ind = ordered_crossover(ind1, ind2, self.random)
        return child_ind.giant_tour

    def _mutate(self, nodes: List[int]) -> List[int]:
        """Mutation via random removal."""
        if not nodes:
            return nodes
        removable = [n for n in nodes if n not in self.mandatory_nodes]
        if not removable:
            return nodes

        n_remove = max(1, int(len(removable) * 0.2))
        to_remove = self.random.sample(removable, min(n_remove, len(removable)))
        return [n for n in nodes if n not in to_remove]

    def _improve(self, nodes: List[int]) -> List[List[int]]:
        """Reconstruct and improve via ALNS-style greedy profit insertion."""
        # Ensure mandatory nodes
        node_set = set(nodes)
        for mn in self.mandatory_nodes:
            node_set.add(mn)

        try:
            # Reconstruct routes
            routes = greedy_profit_insertion(
                [], list(node_set), self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.mandatory_nodes
            )

            # Simple improvement loop
            for _ in range(self.params.alns_iterations):
                # Destroy
                current_nodes = [n for r in routes for n in r]
                removable = [n for n in current_nodes if n not in self.mandatory_nodes]
                if not removable:
                    break

                n_rem = max(1, int(len(removable) * 0.1))
                removed = self.random.sample(removable, min(n_rem, len(removable)))
                remaining = [n for n in current_nodes if n not in removed]

                # Repair using profit-based greedy insertion
                new_routes = greedy_profit_insertion(
                    [],
                    remaining + removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    self.mandatory_nodes,
                )

                if self._evaluate(new_routes) > self._evaluate(routes):
                    routes = new_routes

            return routes
        except Exception:
            return []

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Evaluate net profit."""
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return revenue - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculate total distance."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
