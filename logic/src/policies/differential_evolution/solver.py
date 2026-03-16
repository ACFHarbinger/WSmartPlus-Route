"""
Differential Evolution (DE/rand/1/bin) algorithm for VRPP.

This module implements the rigorous Differential Evolution algorithm as formulated
by Storn & Price (1997), replacing the metaphor-heavy Artificial Bee Colony (ABC)
implementation with proper DE mechanics.

Algorithm:
    1. Initialize population with NP random solution vectors.
    2. For each generation:
        a. Mutation: For each target vector x_i, create mutant v_i = x_r1 + F(x_r2 - x_r3)
        b. Crossover: Create trial vector u_i by binomial crossover between x_i and v_i
        c. Selection: Greedy replacement - keep u_i if f(u_i) ≥ f(x_i), else keep x_i

Key Differences from ABC:
    - DE uses greedy one-to-one selection (not fitness-proportionate)
    - DE has explicit crossover operator with CR parameter
    - DE has no "employed/onlooker/scout" metaphor
    - DE has no trial counter or abandonment mechanism
    - DE mutation uses explicit differential: F × (x_r2 - x_r3)

Reference:
    Storn, R., & Price, K. (1997). "Differential Evolution – A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces."
    Journal of Global Optimization, 11(4), 341-359.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..ant_colony_optimization.k_sparse_aco.params import ACOParams
from ..other.operators import greedy_insertion, random_removal
from .params import DEParams


class DESolver(PolicyVizMixin):
    """
    Differential Evolution (DE/rand/1/bin) solver for VRPP.

    Implements the classical DE algorithm with:
    - Random base vector selection (DE/rand)
    - Single differential mutation vector (DE/rand/1)
    - Binomial crossover (DE/rand/1/bin)
    - Greedy selection
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: DEParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the DE solver.

        Args:
            dist_matrix: Distance matrix of shape (n+1, n+1), index 0 is depot.
            wastes: Mapping of node IDs to waste quantities.
            capacity: Maximum vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            params: DE configuration parameters.
            mandatory_nodes: Nodes that must be included in any solution.
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

        # Pre-instantiate local search for reuse
        from logic.src.policies.other.local_search.local_search_aco import ACOLocalSearch

        aco_params = ACOParams(local_search_iterations=self.params.local_search_iterations)
        self.ls = ACOLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=aco_params,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute the DE/rand/1/bin optimization loop.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).

        Complexity:
            Time: O(G × NP × n²) where G = max_iterations, NP = pop_size
            Space: O(NP × n) to store population
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initialize population of NP solution vectors
        population = [self._initialize_solution() for _ in range(self.params.pop_size)]
        fitness = [self._evaluate(sol) for sol in population]

        # Track global best
        best_idx = int(np.argmax(fitness))
        best_routes = copy.deepcopy(population[best_idx])
        best_profit = fitness[best_idx]
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # For each target vector in population
            for i in range(self.params.pop_size):
                # --- Mutation: v_i = x_r1 + F × (x_r2 - x_r3) ---
                # Select three distinct random indices r1, r2, r3 ≠ i
                candidates = [j for j in range(self.params.pop_size) if j != i]
                if len(candidates) < 3:
                    continue  # Skip if population too small

                r1, r2, r3 = self.rng.sample(candidates, 3)

                # Create mutant vector via differential mutation
                mutant = self._differential_mutation(
                    base=population[r1],
                    diff1=population[r2],
                    diff2=population[r3],
                    F=self.params.mutation_factor,
                )

                # --- Crossover: u_i = binomial_crossover(x_i, v_i, CR) ---
                trial = self._binomial_crossover(target=population[i], mutant=mutant, CR=self.params.crossover_rate)

                # Evaluate trial vector
                trial_fitness = self._evaluate(trial)

                # --- Selection: Greedy replacement ---
                if trial_fitness > fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update global best
                    if trial_fitness > best_profit:
                        best_routes = copy.deepcopy(trial)
                        best_profit = trial_fitness
                        best_cost = self._cost(trial)

            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                pop_size=self.params.pop_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers: DE operators
    # ------------------------------------------------------------------

    def _differential_mutation(
        self,
        base: List[List[int]],
        diff1: List[List[int]],
        diff2: List[List[int]],
        F: float,
    ) -> List[List[int]]:
        """
        DE mutation operator: v = x_r1 + F × (x_r2 - x_r3).

        In discrete routing space, we implement this as:
        1. Extract nodes from diff1 (x_r2) that are NOT in diff2 (x_r3) → differential
        2. Scale by F (probabilistically select F fraction of differential nodes)
        3. Add to base (x_r1) via destroy-repair

        Args:
            base: Base vector (x_r1)
            diff1: First difference vector (x_r2)
            diff2: Second difference vector (x_r3)
            F: Mutation factor (differential weight)

        Returns:
            Mutant vector v_i
        """
        # Extract node sets
        base_nodes = set(node for route in base for node in route)
        diff1_nodes = set(node for route in diff1 for node in route)
        diff2_nodes = set(node for route in diff2 for node in route)

        # Compute differential: nodes in diff1 but not in diff2
        differential = diff1_nodes - diff2_nodes

        # Scale by F (probabilistically select F fraction of differential)
        scaled_differential = {node for node in differential if self.rng.random() < F} if differential else set()

        # Also compute reverse differential for diversity
        reverse_differential = diff2_nodes - diff1_nodes
        scaled_reverse = (
            {node for node in reverse_differential if self.rng.random() < (1.0 - F)} if reverse_differential else set()
        )

        # Combine differentials
        mutation_nodes = scaled_differential | scaled_reverse

        # If no differential, fall back to random perturbation
        if not mutation_nodes:
            mutation_nodes = set(self.rng.sample(self.nodes, min(self.params.n_removal, len(self.nodes))))

        # Apply differential mutation to base
        mutant = copy.deepcopy(base)

        # Remove mutation_nodes from base
        for route in mutant:
            for node in list(route):
                if node in mutation_nodes:
                    route.remove(node)
        mutant = [r for r in mutant if r]

        # Also remove some random nodes for additional perturbation
        try:
            partial, removed = random_removal(mutant, self.params.n_removal, rng=self.rng)
            to_insert = sorted(list(mutation_nodes | set(removed)))

            repaired = greedy_insertion(
                partial,
                to_insert,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )

            # Apply local search to mutant
            return self.ls.optimize(repaired)
        except Exception:
            return copy.deepcopy(base)

    def _binomial_crossover(self, target: List[List[int]], mutant: List[List[int]], CR: float) -> List[List[int]]:
        """
        Binomial crossover operator.

        For each component (node), inherit from mutant with probability CR,
        otherwise inherit from target. At least one component must come from mutant.

        Args:
            target: Target vector (x_i)
            mutant: Mutant vector (v_i)
            CR: Crossover probability

        Returns:
            Trial vector (u_i)
        """
        target_nodes = set(node for route in target for node in route)
        mutant_nodes = set(node for route in mutant for node in route)

        # Binomial crossover: for each node, decide which parent to inherit from
        # j_rand ensures at least one component from mutant
        all_nodes = target_nodes | mutant_nodes
        if not all_nodes:
            return copy.deepcopy(target)

        j_rand = self.rng.choice(list(all_nodes)) if all_nodes else None

        trial_nodes = set()
        for node in all_nodes:
            # Inherit from mutant if rand < CR or this is j_rand
            if self.rng.random() < CR or node == j_rand:
                if node in mutant_nodes:
                    trial_nodes.add(node)
            else:
                # Inherit from target
                if node in target_nodes:
                    trial_nodes.add(node)

        # Reconstruct solution from trial_nodes
        # Use greedy insertion to build feasible routes
        if not trial_nodes:
            return copy.deepcopy(target)

        try:
            trial_routes = greedy_insertion(
                [],
                sorted(list(trial_nodes)),
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
            return trial_routes
        except Exception:
            return copy.deepcopy(target)

    # ------------------------------------------------------------------
    # Private helpers: Solution construction and evaluation
    # ------------------------------------------------------------------

    def _initialize_solution(self) -> List[List[int]]:
        """
        Create initial solution using nearest neighbor heuristic.

        Returns:
            Initial routing solution.
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

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Evaluate fitness (net profit) of a solution.

        Args:
            routes: Routing solution to evaluate

        Returns:
            Net profit = Revenue - Cost
        """
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return revenue - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing distance.

        Args:
            routes: Routing solution

        Returns:
            Total distance traveled
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            # Depot to first node
            total += self.dist_matrix[0][route[0]]
            # Inter-node distances
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            # Last node to depot
            total += self.dist_matrix[route[-1]][0]
        return total
