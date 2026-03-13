"""
Hybrid Memetic Search (HMS) for VRPP.

This solver is functionally identical to the Hybrid Volleyball Premier League
(HVPL) algorithm but uses rigorous nomenclature.

Metaphor Mapping:
- Volleyball Teams → Chromosomes (Routing solutions)
- Players → Route segments/decision variables
- Matches/Competition → Fitness evaluation and Selection
- Coaching → Intensive Local Search (ALNS)
- Passive Teams → Reserve Pool

Algorithm structure:
    1. **Initialization (ACO)**: Construct an initial population (active)
       and a reserve pool (passive) guided by ant colony pheromones.
    2. **Evolution (HGS)**: Apply genetic operators (Crossover and Mutation)
       to evolve the population.
    3. **Selection**: Choose winners for the next generation via elitism.
    4. **Substitution**: Inject fresh diversity by substituting a fraction
       of the active population with chromosomes from the passive pool.
    5. **Coaching (ALNS)**: Apply a full coaching session (Adaptive Large
       Neighborhood Search) to *every* chromosome in the active population.
    6. **Global Update**: Update pheromones using the global best to guide
       future constructions.

Reference:
    Sun, S., Ma, L., Liu, Y., & Wang, L. (2023).
    "Volleyball premier league algorithm with ACO and ALNS for
    simultaneous pickup–delivery location routing problem."
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..adaptive_large_neighborhood_search.alns import ALNSSolver
from ..ant_colony_optimization.k_sparse_aco.solver import KSparseACOSolver
from ..other.operators import greedy_insertion, random_removal
from .params import HybridMemeticSearchParams


class HybridMemeticSearchSolver(PolicyVizMixin):
    """
    Hybrid Memetic Search (HMS) solver for VRPP.

    Functionally equivalent to HVPL.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HybridMemeticSearchParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize Hybrid Memetic Search solver."""
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

        # Phase 1 components: ACO and ALNS
        self.aco_solver = KSparseACOSolver(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=params.aco_params,
            mandatory_nodes=mandatory_nodes,
            seed=seed,
        )

        self.alns_solver = ALNSSolver(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=params.alns_params,
            mandatory_nodes=mandatory_nodes,
            seed=seed,
        )

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute Memetic Island Model Genetic Algorithm.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # ═══════════════════════════════════════════════════════════
        # PHASE 1: ACO-DRIVEN INITIALIZATION
        # ═══════════════════════════════════════════════════════════
        active_pop = self._aco_initialization()
        passive_pool = self._aco_initialization()

        active_fitness = [self._evaluate(sol) for sol in active_pop]

        # Initial sort
        sorted_indices = sorted(range(len(active_fitness)), key=lambda i: active_fitness[i], reverse=True)
        active_pop = [active_pop[i] for i in sorted_indices]
        active_fitness = [active_fitness[i] for i in sorted_indices]

        best_routes = copy.deepcopy(active_pop[0])
        best_profit = active_fitness[0]
        best_cost = self._calculate_cost(best_routes)

        # ═══════════════════════════════════════════════════════════
        # EVOLUTION LOOP
        # ═══════════════════════════════════════════════════════════
        for gen in range(self.params.max_generations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Phase 2: HGS Evolution (Crossover and Mutation)
            offspring = self._hgs_evolution(active_pop, active_fitness)
            offspring_fitness = [self._evaluate(sol) for sol in offspring]

            # Merge and select next generation (Elitism)
            combined_pop = active_pop + offspring
            combined_fitness = active_fitness + offspring_fitness

            active_pop, active_fitness = self._select_next_gen(combined_pop, combined_fitness)

            # Phase 3: Substitution (Inject diversity from passive pool)
            active_pop = self._substitution_phase(active_pop, passive_pool)

            # Phase 4: Coaching (Intensive ALNS Refinement on ALL)
            active_pop = self._alns_coaching(active_pop)
            active_fitness = [self._evaluate(sol) for sol in active_pop]

            # Re-sort after coaching
            sorted_indices = sorted(range(len(active_fitness)), key=lambda i: active_fitness[i], reverse=True)
            active_pop = [active_pop[i] for i in sorted_indices]
            active_fitness = [active_fitness[i] for i in sorted_indices]

            # Update global best and guided construction pheromones
            if active_fitness[0] > best_profit:
                best_routes = copy.deepcopy(active_pop[0])
                best_profit = active_fitness[0]
                best_cost = self._calculate_cost(best_routes)
                self._update_pheromones(best_routes, best_cost)

            self._viz_record(
                iteration=gen,
                best_profit=best_profit,
                best_cost=best_cost,
                avg_profit=np.mean(active_fitness),
                active_size=len(active_pop),
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _aco_initialization(self) -> List[List[List[int]]]:
        """Intelligent population seeding using ant colony guidance."""
        pool = []
        for _ in range(self.params.aco_init_iterations):
            routes = self.aco_solver.constructor.construct()
            if routes:
                pool.append(routes)

        while len(pool) < self.params.population_size:
            pool.append(self._random_construction())

        # Select diverse elite
        return self._select_diverse_elite(pool, self.params.population_size)

    def _random_construction(self) -> List[List[int]]:
        """NN seeding fallback."""
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

    def _select_diverse_elite(self, pool: List[List[List[int]]], n_select: int) -> List[List[List[int]]]:
        """Select top n solutions by quality."""
        if len(pool) <= n_select:
            return pool
        profits = [self._evaluate(sol) for sol in pool]
        sorted_indices = sorted(range(len(profits)), key=lambda i: profits[i], reverse=True)
        return [pool[i] for i in sorted_indices[:n_select]]

    # ------------------------------------------------------------------
    # Evolution Operators
    # ------------------------------------------------------------------

    def _hgs_evolution(self, population: List[List[List[int]]], fitness: List[float]) -> List[List[List[int]]]:
        """Apply HGS-style genetic operations."""
        offspring = []
        for _ in range(len(population)):
            p1 = self._tournament_select(population, fitness)
            p2 = self._tournament_select(population, fitness)

            child = self._crossover(p1, p2) if self.rng.random() < self.params.crossover_rate else copy.deepcopy(p1)

            if self.rng.random() < self.params.mutation_rate:
                child = self._mutate(child)

            offspring.append(child)
        return offspring

    def _tournament_select(self, pop: List[List[List[int]]], fits: List[float], k: int = 3) -> List[List[int]]:
        """Select winner from random pool."""
        pool = self.rng.sample(range(len(pop)), min(k, len(pop)))
        best = max(pool, key=lambda i: fits[i])
        return copy.deepcopy(pop[best])

    def _crossover(self, p1: List[List[int]], p2: List[List[int]]) -> List[List[int]]:
        """Hybrid OX crossover with node-set combination."""
        nodes1 = {n for r in p1 for n in r}
        nodes2 = {n for r in p2 for n in r}
        child_nodes = list(nodes1 | nodes2)

        # Prune if over-sized
        if len(child_nodes) > len(nodes1):
            scored = [(self.wastes.get(n, 0.0), n) for n in child_nodes if n not in self.mandatory_nodes]
            scored.sort(reverse=True)
            keep_count = int(len(nodes1) * 1.1)
            child_nodes = self.mandatory_nodes + [n for _, n in scored[:keep_count]]

        try:
            return greedy_insertion(
                [],
                child_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
        except Exception:
            return copy.deepcopy(p1)

    def _mutate(self, routes: List[List[int]]) -> List[List[int]]:
        """Destroy-repair mutation."""
        try:
            n_node = sum(len(r) for r in routes)
            n_remove = max(2, int(n_node * 0.2))
            partial, removed = random_removal(routes, n_remove, self.rng)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
        except Exception:
            return copy.deepcopy(routes)

    def _select_next_gen(
        self, combined: List[List[List[int]]], fits: List[float]
    ) -> Tuple[List[List[List[int]]], List[float]]:
        """Selection via elitism."""
        sorted_indices = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
        top_indices = sorted_indices[: self.params.population_size]
        return [combined[i] for i in top_indices], [fits[i] for i in top_indices]

    def _substitution_phase(
        self, active: List[List[List[int]]], passive: List[List[List[int]]]
    ) -> List[List[List[int]]]:
        """Inject diversity from reserve pool."""
        n_sub = max(1, int(len(active) * self.params.substitution_rate))
        for i in range(len(active) - n_sub, len(active)):
            active[i] = copy.deepcopy(self.rng.choice(passive))
        return active

    def _alns_coaching(self, population: List[List[List[int]]]) -> List[List[List[int]]]:
        """Intensive refinement phase (ALNS coaching)."""
        coached = []
        for individual in population:
            refined, _, _ = self.alns_solver.solve(initial_solution=individual)
            coached.append(refined)
        return coached

    # ------------------------------------------------------------------
    # Pheromone Guidance
    # ------------------------------------------------------------------

    def _update_pheromones(self, best_routes: List[List[int]], best_cost: float) -> None:
        """Global pheromone reinforcement."""
        if not best_routes or best_cost <= 0:
            return
        self.aco_solver.pheromone.evaporate_all(self.params.aco_params.rho)
        delta = 1.0 / best_cost
        for r in best_routes:
            if not r:
                continue
            self.aco_solver.pheromone.update_edge(0, r[0], delta, evaporate=False)
            for k in range(len(r) - 1):
                self.aco_solver.pheromone.update_edge(r[k], r[k + 1], delta, evaporate=False)
            self.aco_solver.pheromone.update_edge(r[-1], 0, delta, evaporate=False)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Calculate net profit."""
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._calculate_cost(routes) * self.C

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total distance."""
        total = 0.0
        for r in routes:
            if not r:
                continue
            total += self.dist_matrix[0][r[0]]
            for k in range(len(r) - 1):
                total += self.dist_matrix[r[k]][r[k + 1]]
            total += self.dist_matrix[r[-1]][0]
        return total
