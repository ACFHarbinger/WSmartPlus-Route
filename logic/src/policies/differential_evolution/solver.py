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

from logic.src.policies.other.local_search.local_search_aco import ACOLocalSearch

from ..ant_colony_optimization_k_sparse.params import KSACOParams
from ..other.operators import greedy_insertion, greedy_profit_insertion
from ..other.operators.heuristics.greedy_initialization import build_greedy_routes
from .params import DEParams


class DESolver:
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
        self.rng = random.Random(params.seed) if params.seed is not None else random.Random(42)

        # Pre-instantiate local search for reuse
        aco_params = KSACOParams(
            local_search_iterations=self.params.local_search_iterations,
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

            getattr(self, "_viz_record", lambda **k: None)(
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
        DE mutation operator: v = x_r1 + F * (x_r2 - x_r3).

        Discrete mapping using weighted set symmetric differences:
        1. Identify nodes present in diff1 but NOT in diff2 (positive differential).
        2. Identify nodes present in diff2 but NOT in diff1 (negative differential).
        3. Probabilistically (factor F) extract nodes from the positive differential
           to add to the base, and from the negative differential to remove from the base.
        4. This simulates a 'direction' in the discrete search space.
        """
        base_nodes = set(n for r in base for n in r)
        diff1_nodes = set(n for r in diff1 for n in r)
        diff2_nodes = set(n for r in diff2 for n in r)

        # Vector differential: d = (x_r2 - x_r3)
        pos_diff = diff1_nodes - diff2_nodes
        neg_diff = diff2_nodes - diff1_nodes

        # Scaled differential: F * d
        to_add = {n for n in pos_diff if self.rng.random() < F}
        to_remove = {n for n in neg_diff if self.rng.random() < F}

        # Vector addition: v = x_r1 + F * d
        mutant_nodes = (base_nodes - to_remove) | to_add

        if not mutant_nodes:
            # Ensure the mutant is not empty if the problem has mandatory nodes
            if self.mandatory_nodes:
                mutant_nodes = set(self.mandatory_nodes)
            else:
                return copy.deepcopy(base)

        # Build solution using greedy insertion for feasibility
        try:
            if self.params.profit_aware_operators:
                mutant = greedy_profit_insertion(
                    [],
                    sorted(list(mutant_nodes)),
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            else:
                mutant = greedy_insertion(
                    [],
                    sorted(list(mutant_nodes)),
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            # Local search is an extension common in VRP-DE
            return self.ls.optimize(mutant)
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
            # Inherit from mutant if node is j_rand OR rand < CR
            if node == j_rand or self.rng.random() < CR:
                # If inherited from mutant, the node is in trial if it's in mutant_nodes
                if node in mutant_nodes:
                    trial_nodes.add(node)
                # Note: if node == j_rand and node not in mutant_nodes, we don't add it.
                # This correctly reflects the mutant vector's state for that "dimension".
            else:
                # Inherit from target
                if node in target_nodes:
                    trial_nodes.add(node)

        # Reconstruct solution from trial_nodes
        # Use greedy insertion to build feasible routes
        if not trial_nodes:
            return copy.deepcopy(target)

        try:
            if self.params.profit_aware_operators:
                trial_routes = greedy_profit_insertion(
                    [],
                    sorted(list(trial_nodes)),
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            else:
                trial_routes = greedy_insertion(
                    [],
                    sorted(list(trial_nodes)),
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            return trial_routes
        except Exception:
            return copy.deepcopy(target)

    # ------------------------------------------------------------------
    # Private helpers: Solution construction and evaluation
    # ------------------------------------------------------------------

    def _initialize_solution(self) -> List[List[int]]:
        """
        Create initial solution using greedy constructive heuristic.

        Returns:
            Initial routing solution.
        """
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.rng,
        )

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
