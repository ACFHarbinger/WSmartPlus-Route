"""
Hybrid Genetic Search (HGS) policy module.

Combines genetic algorithms with local search and the Split algorithm
for solving the Capacitated Vehicle Routing Problem with Profits.

Reference:
    Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source
    implementation and SWAP* neighborhood. Computers & Operations Research,
    140, 105643.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.other.operators.crossover.ordered import ordered_crossover

from ..other.local_search.local_search_hgs import HGSLocalSearch
from .evolution import evaluate, update_biased_fitness
from .individual import Individual
from .params import HGSParams
from .split import LinearSplit


class HGSSolver:
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
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
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
            x_coords: Optional array of x-coordinates indexed by node id (index 0 = depot).
                      When provided together with y_coords, enables SWAP* polar sector pruning.
            y_coords: Optional array of y-coordinates indexed by node id (index 0 = depot).
                      When provided together with x_coords, enables SWAP* polar sector pruning.
        """
        self.d = dist_matrix
        self.wastes = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(params.seed) if params.seed is not None else random.Random(42)

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        self.split_manager = LinearSplit(
            dist_matrix, wastes, capacity, R, C, params.max_vehicles, mandatory_nodes, vrpp=params.vrpp
        )
        self.ls = HGSLocalSearch(dist_matrix, wastes, capacity, R, C, params)
        if x_coords is not None and y_coords is not None:
            self.ls.x_coords = x_coords
            self.ls.y_coords = y_coords

    def _initialize_population(self, penalty_capacity: float) -> Tuple[List[Individual], List[Individual]]:
        """
        Initialize dual subpopulations with random solutions improved by local search.

        Args:
            penalty_capacity: Initial penalty coefficient for capacity violations.

        Returns:
            Tuple of (feasible population, infeasible population).
        """
        pop_feasible: List[Individual] = []
        pop_infeasible: List[Individual] = []

        initial_size = 4 * self.params.mu
        for _ in range(initial_size):
            gt = self.nodes[:]
            self.random.shuffle(gt)
            ind = Individual(gt)
            evaluate(ind, self.split_manager, penalty_capacity)

            # Educate with local search
            ind = self.ls.optimize(ind)
            evaluate(ind, self.split_manager, penalty_capacity)

            # Insert into appropriate subpopulation
            if ind.is_feasible:
                pop_feasible.append(ind)
            else:
                pop_infeasible.append(ind)

        # Initialize fitness for both subpopulations
        update_biased_fitness(pop_feasible, self.params.nb_elite, self.params.nb_close)
        update_biased_fitness(pop_infeasible, self.params.nb_elite, self.params.nb_close)

        return pop_feasible, pop_infeasible

    def _trim_populations(self, pop_feasible: List[Individual], pop_infeasible: List[Individual]) -> None:
        self._trim_one(pop_feasible)
        self._trim_one(pop_infeasible)

    def _trim_one(self, pop: List[Individual]) -> None:
        """Trim one sub-population to mu, removing clones first then worst by fitness."""
        max_pop_size = self.params.mu + self.params.lambda_param
        if len(pop) <= max_pop_size:
            return

        update_biased_fitness(pop, self.params.nb_elite, self.params.nb_close)

        while len(pop) > self.params.mu:
            # Step 1: remove a clone (broken-pairs distance == 0) if one exists
            clone_idx = self._find_clone(pop)
            if clone_idx is not None:
                pop.pop(clone_idx)
                continue
            # Step 2: no clone — remove the individual with the worst (highest) fitness
            worst_idx = max(range(len(pop)), key=lambda i: pop[i].fitness)
            pop.pop(worst_idx)

    def _find_clone(self, pop: List[Individual]) -> Optional[int]:
        """Return the index of any individual whose routes are identical to another's, or None."""
        seen: set = set()
        for i, ind in enumerate(pop):
            # Use a frozenset of tuples as a hashable route signature
            key = frozenset(tuple(r) for r in ind.routes)
            if key in seen:
                return i
            seen.add(key)
        return None

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Hybrid Genetic Search algorithm following Vidal et al. (2022).

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        start_time = time.process_time()

        # Initialize dual subpopulations and penalty
        penalty_capacity = self.params.initial_penalty_capacity
        pop_feasible, pop_infeasible = self._initialize_population(penalty_capacity)
        it = 0
        it_no_improvement = 0
        best_profit_so_far = max(ind.profit_score for ind in pop_feasible) if pop_feasible else -float("inf")

        # Main evolutionary loop (Algorithm 1)
        while it_no_improvement < self.params.n_iterations_no_improvement:
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break
            it += 1
            it_no_improvement += 1

            # Generate and educate offspring
            combined_pop = pop_feasible + pop_infeasible
            if len(combined_pop) < 2:
                break

            child = self._generate_offspring(combined_pop, penalty_capacity)

            # Insert into subpopulation and optionally repair
            self._insert_and_repair(child, pop_feasible, pop_infeasible, penalty_capacity)

            # Track improvements
            if child.is_feasible and child.profit_score > best_profit_so_far:
                best_profit_so_far = child.profit_score
                it_no_improvement = 0

            # Trim populations and adjust penalties (batched every 100 iterations per Vidal 2022)
            self._trim_populations(pop_feasible, pop_infeasible)
            if it % 100 == 0:
                penalty_capacity = self._adjust_penalties(pop_feasible, pop_infeasible, penalty_capacity)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=it,
                best_profit=best_profit_so_far,
                child_profit=child.profit_score,
                child_cost=child.cost,
                population_size=len(pop_feasible) + len(pop_infeasible),
            )

        return self._get_best_solution(pop_feasible, pop_infeasible)

    def _generate_offspring(self, population: List[Individual], penalty_capacity: float) -> Individual:
        """Generate and educate offspring via crossover and local search."""
        p1, p2 = self._select_parents(population)
        child = ordered_crossover(p1, p2, rng=self.random)
        evaluate(child, self.split_manager, penalty_capacity)
        child = self.ls.optimize(child)
        evaluate(child, self.split_manager, penalty_capacity)
        return child

    def _insert_and_repair(
        self,
        child: Individual,
        pop_feasible: List[Individual],
        pop_infeasible: List[Individual],
        penalty_capacity: float,
    ) -> None:
        """Insert child into subpopulation and optionally repair if infeasible."""
        if child.is_feasible:
            pop_feasible.append(child)
        else:
            pop_infeasible.append(child)
            # Repair with 50% probability
            if self.random.random() < self.params.repair_probability:
                repaired = Individual(child.giant_tour[:])
                evaluate(repaired, self.split_manager, penalty_capacity * 10.0)
                repaired = self.ls.optimize(repaired)
                evaluate(repaired, self.split_manager, penalty_capacity)
                if repaired.is_feasible:
                    pop_feasible.append(repaired)

    def _get_best_solution(
        self, pop_feasible: List[Individual], pop_infeasible: List[Individual]
    ) -> Tuple[List[List[int]], float, float]:
        """Return best feasible solution or best infeasible if none exists."""
        if pop_feasible:
            best_ind = max(pop_feasible, key=lambda x: x.profit_score)
        elif pop_infeasible:
            best_ind = max(pop_infeasible, key=lambda x: x.profit_score)
        else:
            return [], 0.0, 0.0
        return best_ind.routes, best_ind.profit_score, best_ind.cost

    def _adjust_penalties(
        self, pop_feasible: List[Individual], pop_infeasible: List[Individual], current_penalty: float
    ) -> float:
        """
        Adjust penalty coefficients to maintain target proportion of feasible solutions.
        Following Vidal et al. (2022), targets ~20% feasible solutions.
        Called every 100 iterations as specified in the paper.

        Args:
            pop_feasible: Feasible subpopulation.
            pop_infeasible: Infeasible subpopulation.
            current_penalty: Current penalty coefficient.

        Returns:
            Updated penalty coefficient.
        """
        total_pop = len(pop_feasible) + len(pop_infeasible)
        if total_pop == 0:
            return current_penalty

        feasible_ratio = len(pop_feasible) / total_pop

        # If too many feasible solutions, decrease penalty (make it easier to violate)
        if feasible_ratio > self.params.target_feasible + 0.05:
            return current_penalty * self.params.penalty_decrease
        # If too many infeasible solutions, increase penalty
        elif feasible_ratio < self.params.target_feasible - 0.05:
            return current_penalty * self.params.penalty_increase
        else:
            return current_penalty

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """
        Select two parents using binary tournament selection.

        Args:
            population: Combined population (feasible + infeasible).

        Returns:
            Two selected parent individuals.
        """

        def tournament():
            """Perform a binary tournament selection."""
            i1, i2 = self.random.sample(population, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()
