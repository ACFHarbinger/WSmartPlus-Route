"""
Memetic Island Model Genetic Algorithm for VRPP.

This solver is functionally identical to the Soccer League Competition (SLC)
algorithm but uses standard Genetic Algorithm and Evolutionary terminology.

Algorithm structure (Island Model):
    1. **Initialization**: Construct multiple islands (sub-populations).
    2. **Intra-Island Competition**: Apply perturbation and local search
       to each individual in the island.
    3. **Inter-Island Competition**: Recombine winners from different
       islands to share genetic material.
    4. **Regeneration**: If an island stagnates, regenerate its population.

Algorithm structure:
    1. Initialize K islands with N chromosomes each.
    2. For each generation:
        a. **Intra-island Evolution**: Each chromosome is perturbed and
           refined via local search (ACOLocalSearch).
        b. **Inter-island Evolution**: Islands are paired for "matches".
           A stochastic recombination occurs where the "losing" island's
           weakest chromosome adopts structure from the "winning" island's
           best chromosome.
        c. **Island Regeneration**: If an island fails to improve for
           `stagnation_limit` generations, it is fully randomized.
        d. **Global best update**: Track the overall best solution.

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

from logic.src.policies.other.local_search.local_search_aco import ACOLocalSearch
from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..other.operators import greedy_insertion, random_removal
from .params import MemeticAlgorithmIslandModelParams


class MemeticAlgorithmIslandModelSolver(PolicyVizMixin):
    """
    Memetic Algorithm with Island Model (MA-IM) solver for VRPP.

    Functionally equivalent to SLC but with rigorous nomenclature.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MemeticAlgorithmIslandModelParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize Memetic Island Model GA solver."""
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
        Execute Memetic Island Model Genetic Algorithm.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # Phase 1: Initialize K islands with N chromosomes
        islands: List[List[Tuple[List[List[int]], float]]] = [
            self._initialize_island() for _ in range(self.params.n_islands)
        ]
        stagnation_counters: List[int] = [0] * self.params.n_islands
        island_best_fitness: List[float] = [max(f for _, f in island) for island in islands]

        # Track global best
        best_routes, best_profit = self._get_global_best(islands)
        best_cost = self._calculate_cost(best_routes)

        # Phase 2: Evolution Loop
        for generation in range(self.params.max_generations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Phase 2a: Intra-island Evolution (Perturbation + Local Search)
            for _i_idx, island in enumerate(islands):
                for c_idx in range(len(island)):
                    routes, fitness = island[c_idx]
                    perturbed_routes = self._perturb(routes)
                    perturbed_fitness = self._evaluate_fitness(perturbed_routes)
                    if perturbed_fitness > fitness:
                        island[c_idx] = (perturbed_routes, perturbed_fitness)

            # Phase 2b: Inter-island Evolution (Stochastic Recombination / "Matches")
            island_indices = list(range(self.params.n_islands))
            self.rng.shuffle(island_indices)

            for k in range(0, len(island_indices) - 1, 2):
                idx_a = island_indices[k]
                idx_b = island_indices[k + 1]

                fitness_a = float(sum(f for _, f in islands[idx_a]))
                fitness_b = float(sum(f for _, f in islands[idx_b]))

                total_fitness = abs(fitness_a) + abs(fitness_b) + 1e-9
                win_prob_a = (fitness_a - min(fitness_a, fitness_b) + 1e-9) / total_fitness

                if self.rng.random() < win_prob_a:
                    winner_idx, loser_idx = idx_a, idx_b
                else:
                    winner_idx, loser_idx = idx_b, idx_a

                # Weakest chromosome in loser island adopts structure from winner's best
                winner_best_routes = max(islands[winner_idx], key=lambda x: x[1])[0]
                loser_worst_idx = int(np.argmin([f for _, f in islands[loser_idx]]))

                recombined_child = self._recombine(islands[loser_idx][loser_worst_idx][0], winner_best_routes)
                recombined_fitness = self._evaluate_fitness(recombined_child)
                islands[loser_idx][loser_worst_idx] = (recombined_child, recombined_fitness)

            # Phase 2c: Stagnation Check and Island Regeneration
            for i_idx, island in enumerate(islands):
                current_island_best = max(f for _, f in island)
                if current_island_best > island_best_fitness[i_idx] + 1e-9:
                    island_best_fitness[i_idx] = current_island_best
                    stagnation_counters[i_idx] = 0
                else:
                    stagnation_counters[i_idx] += 1
                    if stagnation_counters[i_idx] >= self.params.stagnation_limit:
                        islands[i_idx] = self._initialize_island()
                        stagnation_counters[i_idx] = 0
                        island_best_fitness[i_idx] = max(f for _, f in islands[i_idx])

            # Update global best
            gen_best_routes, gen_best_fitness = self._get_global_best(islands)
            if gen_best_fitness > best_profit:
                best_routes = copy.deepcopy(gen_best_routes)
                best_profit = gen_best_fitness
                best_cost = self._calculate_cost(best_routes)

            self._viz_record(
                iteration=generation,
                best_profit=best_profit,
                best_cost=best_cost,
                n_islands=self.params.n_islands,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Evolution Operators
    # ------------------------------------------------------------------

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """Intra-island perturbation with intensive local search."""
        n_remove = max(3, self.params.n_removal)
        try:
            partial, removed = random_removal(routes, n_remove, self.rng)
            repaired = greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            # Apply intensive local search refinement (Memetic component)
            ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
            return ls.optimize(repaired)
        except Exception:
            return copy.deepcopy(routes)

    def _recombine(self, loser_routes: List[List[int]], winner_routes: List[List[int]]) -> List[List[int]]:
        """Inter-island recombination (stochastic match logic)."""
        winner_flat = [n for r in winner_routes for n in r]
        loser_flat = [n for r in loser_routes for n in r]

        if len(winner_flat) < 2:
            return copy.deepcopy(loser_routes)

        # Extract segments for recombination
        a = self.rng.randint(0, len(winner_flat) - 1)
        b = self.rng.randint(a, min(a + max(1, len(winner_flat) // 3), len(winner_flat)))
        segment = winner_flat[a:b]
        segment_set = set(segment)

        # Order-preserving recombination
        remaining = [n for n in loser_flat if n not in segment_set]
        insert_pos = min(a, len(remaining))
        child_flat = remaining[:insert_pos] + segment + remaining[insert_pos:]

        # Split into routes by capacity
        child_routes: List[List[int]] = []
        curr_route: List[int] = []
        load = 0.0
        for node in child_flat:
            waste = self.wastes.get(node, 0.0)
            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    child_routes.append(curr_route)
                curr_route = [node]
                load = waste
        if curr_route:
            child_routes.append(curr_route)

        # Ensure mandatory nodes
        visited = {n for r in child_routes for n in r}
        for n in self.mandatory_nodes:
            if n not in visited:
                child_routes.append([n])

        # Apply intensive local search refinement (Memetic component)
        ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
        return ls.optimize(child_routes)

    # ------------------------------------------------------------------
    # Initialization and Utilities
    # ------------------------------------------------------------------

    def _initialize_island(self) -> List[Tuple[List[List[int]], float]]:
        """Create a fresh island of randomized chromosomes."""
        island = []
        for _ in range(self.params.island_size):
            routes = self._build_random_solution()
            fitness = self._evaluate_fitness(routes)
            island.append((routes, fitness))
        return island

    def _build_random_solution(self) -> List[List[int]]:
        """Build initial solution using nearest-neighbor heuristic."""
        from logic.src.policies.other.operators.heuristics.nn_initialization import build_nn_routes

        return build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.rng,
        )

    def _get_global_best(self, islands: List[List[Tuple[List[List[int]], float]]]) -> Tuple[List[List[int]], float]:
        """Find global best chromosome across all islands."""
        best_p = -float("inf")
        best_r: List[List[int]] = []
        for island in islands:
            for routes, fitness in island:
                if fitness > best_p:
                    best_p = fitness
                    best_r = routes
        return copy.deepcopy(best_r), best_p

    def _evaluate_fitness(self, routes: List[List[int]]) -> float:
        """Calculate net profit for a routing solution."""
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return revenue - self._calculate_cost(routes) * self.C

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total distance traveled."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
