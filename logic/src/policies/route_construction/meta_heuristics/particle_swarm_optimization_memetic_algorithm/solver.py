"""
Particle Swarm Optimization Memetic Algorithm (PSOMA) for VRPP.

Particles navigate the discrete routing space via swap-based velocity.
A low inertia weight (ω≈0.4) forces intensive local exploitation.  A
periodic memetic step applies worst-removal + greedy-insertion local
search to each particle, analogous to the genetic operators described in
the survey.

Attributes:
    PSOMAsSolver (Type): Core solver class for PSOMA.
    PSOMAParams (Type): Parameter dataclass for the solver.

Example:
    >>> solver = PSOMAsSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()

Reference:
    Liu, B., Wang, L., Jin, Y., & Huang, D. (2006).
    "An Effective PSO-Based Memetic Algorithm for TSP"
"""

import contextlib
import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_aco import ACOLocalSearch
from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams

from .params import PSOMAParams
from .particle import PSOMAParticle


class PSOMAsSolver:
    """
    PSOMA solver for VRPP — PSO with memetic local-search steps.

    Attributes:
        dist_matrix (np.ndarray): Symmetric distance matrix.
        wastes (Dict[int, float]): Mapping of bin IDs to waste quantities.
        capacity (float): Maximum vehicle collection capacity.
        R (float): Revenue per kg of waste.
        C (float): Cost per km traveled.
        params (PSOMAParams): Algorithm-specific parameters.
        mandatory_nodes (List[int]): Nodes that must be visited.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: PSOMAParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """Initializes the PSOMA solver.

        Args:
            dist_matrix (np.ndarray): Symmetric distance matrix.
            wastes (Dict[int, float]): Mapping of bin IDs to waste quantities.
            capacity (float): Maximum vehicle collection capacity.
            R (float): Revenue per kg of waste.
            C (float): Cost per km traveled.
            params (PSOMAParams): Algorithm-specific parameters.
            mandatory_nodes (Optional[List[int]]): Nodes that must be visited.
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
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        # Initialize Local Search once for reuse
        aco_params = KSACOParams(
            local_search_iterations=self.params.local_search_iterations,
            time_limit=self.params.time_limit,
            vrpp=self.params.vrpp,
            profit_aware_operators=self.params.profit_aware_operators,
            seed=self.params.seed,
        )

        self.ls = ACOLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=aco_params,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run PSOMA optimisation.

        Returns:
            Tuple[List[List[int]], float, float]: Optimized (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initialise swarm
        swarm = self._init_swarm()
        gbest_routes, gbest_profit = self._global_best(swarm)
        gbest_cost = self._cost(gbest_routes)

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            for particle in swarm:
                # Velocity / position update via probabilistic segment adoption
                particle.routes = self._update_position(
                    particle.routes,
                    particle.pbest_routes,
                    gbest_routes,
                )
                particle.profit = self._evaluate(particle.routes)

                # Update personal best
                if particle.profit > particle.pbest_profit:
                    particle.pbest_routes = copy.deepcopy(particle.routes)
                    particle.pbest_profit = particle.profit

            # Global best update
            for particle in swarm:
                if particle.profit > gbest_profit:
                    gbest_routes = copy.deepcopy(particle.routes)
                    gbest_profit = particle.profit
                    gbest_cost = self._cost(gbest_routes)

            # Memetic step: periodic local search on every particle
            if (iteration + 1) % self.params.local_search_freq == 0:
                for particle in swarm:
                    ls_routes = self._local_search(particle.routes)
                    ls_profit = self._evaluate(ls_routes)
                    if ls_profit > particle.profit:
                        particle.routes = ls_routes
                        particle.profit = ls_profit
                        if ls_profit > particle.pbest_profit:
                            particle.pbest_routes = copy.deepcopy(ls_routes)
                            particle.pbest_profit = ls_profit
                        if ls_profit > gbest_profit:
                            gbest_routes = copy.deepcopy(ls_routes)
                            gbest_profit = ls_profit
                            gbest_cost = self._cost(gbest_routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=gbest_profit,
                best_cost=gbest_cost,
                swarm_size=len(swarm),
            )

        return gbest_routes, gbest_profit, gbest_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_swarm(self) -> List[PSOMAParticle]:
        """Initialise swarm with random feasible solutions.

        Returns:
            List[PSOMAParticle]: Initialized swarm of particles.
        """
        swarm = []
        for _ in range(self.params.pop_size):
            routes = self._build_random_solution()
            profit = self._evaluate(routes)
            swarm.append(PSOMAParticle(routes, profit))
        return swarm

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style).

        Returns:
            List[List[int]]: Feasible initial routes built via nearest-neighbor heuristic.
        """
        optimized_routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.random,
        )
        return optimized_routes

    def _global_best(self, swarm: List[PSOMAParticle]) -> Tuple[List[List[int]], float]:
        """Return routes and profit of the best particle in the swarm.

        Args:
            swarm: List of current particles.

        Returns:
            Tuple[List[List[int]], float]: Deep copy of best routes and their profit.
        """
        best = max(swarm, key=lambda p: p.profit)
        return copy.deepcopy(best.routes), best.profit

    def _update_position(
        self,
        current: List[List[int]],
        pbest: List[List[int]],
        gbest: List[List[int]],
    ) -> List[List[int]]:
        """Update particle position via Swap-Based Velocity (Liu et al. 2006).

        Args:
            current: Current particle routing solution.
            pbest: Particle's personal best routing solution.
            gbest: Global best routing solution across all particles.

        Returns:
            List[List[int]]: Updated routing solution after velocity application and local search.
        """
        routes = copy.deepcopy(current)

        # 1. Inertia: with probability (1 - omega), perform a random move
        if self.random.random() > self.params.omega:
            routes = self._random_relocate(routes)

        # 2. Cognitive: Move toward personal best using swap sequences
        if self.random.random() < self.params.c1 and pbest:
            routes = self._apply_velocity(routes, pbest)

        # 3. Social: Move toward global best using swap sequences
        if self.random.random() < self.params.c2 and gbest:
            routes = self._apply_velocity(routes, gbest)

        # 2-opt local search for refinement (meme)
        return self.ls.optimize(routes)

    def _apply_velocity(self, current: List[List[int]], target: List[List[int]]) -> List[List[int]]:
        """Apply velocity by moving current solution toward target via segment adoption.

        In discrete PSO for VRP, this means adopting segments or performing swaps
        that reduce the distance to the target solution.

        Args:
            current: Current routing solution to be updated.
            target: Target routing solution (personal best or global best).

        Returns:
            List[List[int]]: Updated routing solution after segment adoption.
        """
        curr_flat = [n for r in current for n in r]
        targ_flat = [n for r in target for n in r]

        if not curr_flat or not targ_flat:
            return current

        # Simplified Velocity: adoption of target segments (standard for discrete PSO VRP)
        if len(targ_flat) > 2:
            a = self.random.randint(0, len(targ_flat) - 1)
            b = self.random.randint(a, min(a + 5, len(targ_flat)))
            segment = targ_flat[a:b]
            segment_set = set(segment)

            # Reconstruct preserving target order for that segment
            remaining = [n for n in curr_flat if n not in segment_set]
            insert_pos = min(a, len(remaining))
            new_flat = remaining[:insert_pos] + segment + remaining[insert_pos:]

            # Re-partition into routes based on capacity
            return self._partition_flat(new_flat)

        return current

    def _partition_flat(self, flat_nodes: List[int]) -> List[List[int]]:
        """Partition flattened nodes into capacity-feasible routes.

        Args:
            flat_nodes: Ordered sequence of customer node indices.

        Returns:
            List[List[int]]: Routes respecting vehicle capacity constraints.
        """
        routes: List[List[int]] = []
        curr_route: List[int] = []
        load = 0.0
        for node in flat_nodes:
            waste = self.wastes.get(node, 0.0)
            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    routes.append(curr_route)
                curr_route = [node]
                load = waste
        if curr_route:
            routes.append(curr_route)

        # Ensure mandatory nodes
        visited = {n for r in routes for n in r}
        for n in self.mandatory_nodes:
            if n not in visited:
                routes.append([n])
        return routes

    def _random_relocate(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Randomly remove one node and reinsert it at a random position.

        Args:
            routes: Current routes.

        Returns:
            Perturbed routes.
        """
        if not routes:
            return routes

        new_routes, nodes = random_removal(routes, 1, self.random)
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp
        with contextlib.suppress(Exception):
            if use_profit:
                new_routes = greedy_profit_insertion(
                    new_routes,
                    nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            else:
                new_routes = greedy_insertion(
                    new_routes,
                    nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
        return new_routes

    def _local_search(self, routes: List[List[int]]) -> List[List[int]]:
        """Apply memetic local search: worst-removal + greedy-insertion + ACO.

        Args:
            routes: Current routing solution to improve.

        Returns:
            List[List[int]]: Locally improved routing solution.
        """
        n = max(3, self.params.n_removal)
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        try:
            if use_profit:
                partial, removed = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R)
                repaired = greedy_profit_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            else:
                partial, removed = worst_removal(routes, n, self.dist_matrix)
                repaired = greedy_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            return self.ls.optimize(repaired)
        except Exception:
            return copy.deepcopy(routes)

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Compute net profit for a set of routes.

        Args:
            routes: Routing solution to evaluate.

        Returns:
            float: Net profit (revenue minus travel cost).
        """
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Compute total routing distance across all routes.

        Args:
            routes: Routing solution to evaluate.

        Returns:
            float: Sum of all inter-node distances including depot returns.
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
