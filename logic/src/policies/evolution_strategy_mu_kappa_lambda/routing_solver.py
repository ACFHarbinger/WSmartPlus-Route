"""
(μ,κ,λ) Evolution Strategy adapted for Vehicle Routing Problems.

Adapts the classical continuous ES to work with discrete routing solutions,
maintaining age-based selection mechanism.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..other.operators import greedy_insertion, random_removal
from .params import MuKappaLambdaESParams


class RoutingIndividual:
    """
    Individual for routing ES with age tracking.

    Attributes:
        routes: Routing solution (list of routes).
        fitness: Net profit (higher is better).
        age: Number of generations survived.
    """

    def __init__(self, routes: List[List[int]], fitness: float = -np.inf, age: int = 0):
        """Initialize a routing individual.

        Args:
            routes: List of routes (each route is list of node IDs).
            fitness: Fitness value (net profit).
            age: Age in generations.
        """
        self.routes = copy.deepcopy(routes)
        self.fitness = fitness
        self.age = age

    def copy(self) -> "RoutingIndividual":
        """Create a deep copy."""
        return RoutingIndividual(
            routes=copy.deepcopy(self.routes),
            fitness=self.fitness,
            age=self.age,
        )


class MuKappaLambdaESRoutingSolver(PolicyVizMixin):
    """
    (μ,κ,λ) Evolution Strategy for Vehicle Routing Problems.

    Adapts classical ES with age-based selection to routing domain.
    Parents exceeding age κ are excluded from selection pool.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MuKappaLambdaESParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the (μ,κ,λ)-ES routing solver.

        Args:
            dist_matrix: Distance matrix [n+1 × n+1] including depot at index 0.
            wastes: Dictionary mapping node IDs to waste quantities.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            params: ES configuration parameters.
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

        # Statistics
        self.n_evaluations = 0
        self.best_individual: Optional[RoutingIndividual] = None

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute (μ,κ,λ)-ES for routing.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initialize population P_0 with μ parents
        parents = [self._initialize_individual() for _ in range(self.params.mu)]

        # Evaluate initial population
        for ind in parents:
            ind.fitness = self._evaluate(ind.routes)

        # Track global best
        self._update_best(parents)

        # Evolution loop
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Step 1: Generate λ offspring via recombination and mutation
            offspring = []
            for _ in range(self.params.lambda_):
                # Select ρ parents for recombination
                selected_parents = self._select_parents_for_recombination(parents)

                # Create offspring via recombination
                child = self._recombine(selected_parents)

                # Apply mutation (destroy-repair)
                mutated_child = self._mutate(child)

                # Evaluate offspring
                mutated_child.fitness = self._evaluate(mutated_child.routes)

                offspring.append(mutated_child)

            # Step 2: Age-based selection
            # Eligible parents are those with age ≤ κ
            eligible_parents = [p for p in parents if p.age < self.params.kappa]

            # Combine eligible parents and offspring for selection pool
            selection_pool = eligible_parents + offspring

            # Step 3: Select μ best individuals
            selection_pool.sort(key=lambda ind: ind.fitness, reverse=True)  # Maximize profit
            parents = selection_pool[: self.params.mu]

            # Step 4: Increment age of surviving parents (not offspring)
            for p in parents:
                # Check if this individual was a parent (not a new offspring)
                if p in eligible_parents:
                    p.age += 1
                # New offspring have age = 0 (already set in constructor)

            # Update global best
            self._update_best(parents)

            # Visualization callback
            if self.best_individual:
                self._viz_record(
                    iteration=iteration,
                    best_profit=self.best_individual.fitness,
                    best_cost=self._cost(self.best_individual.routes),
                    population_size=self.params.mu,
                )

        # Return best solution
        if self.best_individual:
            return (
                self.best_individual.routes,
                self.best_individual.fitness,
                self._cost(self.best_individual.routes),
            )
        else:
            return [], 0.0, 0.0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_individual(self) -> RoutingIndividual:
        """
        Initialize a single individual using nearest-neighbor heuristic.

        Returns:
            A routing individual with age = 0.
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

        return RoutingIndividual(routes=routes, age=0)

    # ------------------------------------------------------------------
    # Recombination
    # ------------------------------------------------------------------

    def _select_parents_for_recombination(self, parents: List[RoutingIndividual]) -> List[RoutingIndividual]:
        """
        Select ρ parents for recombination via uniform random sampling.

        Args:
            parents: Current parent population.

        Returns:
            List of ρ randomly selected parents.
        """
        indices = self.rng.sample(range(len(parents)), self.params.rho)
        return [parents[i] for i in indices]

    def _recombine(self, parents: List[RoutingIndividual]) -> RoutingIndividual:
        """
        Create offspring via route crossover.

        For routing problems, we use route-based recombination:
        - Intermediate: Merge routes from all parents, then re-cluster
        - Discrete: Randomly select routes from different parents

        Args:
            parents: List of ρ parent individuals.

        Returns:
            Offspring individual (age = 0).
        """
        if self.params.recombination_type == "intermediate":
            # Collect all nodes from all parents
            all_nodes = set()
            for parent in parents:
                for route in parent.routes:
                    all_nodes.update(route)

            # Build new solution from merged node set
            if all_nodes:
                routes = greedy_insertion(
                    partial_routes=[],
                    nodes_to_insert=sorted(list(all_nodes)),
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    R=self.R,
                    mandatory_nodes=self.mandatory_nodes,
                )
            else:
                routes = []

        else:  # discrete recombination
            # Randomly select routes from parents
            all_routes = []
            for parent in parents:
                all_routes.extend(parent.routes)

            if all_routes:
                # Sample routes randomly
                n_routes = min(len(all_routes), self.params.mu // 2)
                selected_routes = self.rng.sample(all_routes, n_routes)

                # Remove duplicates
                visited = set()
                unique_routes = []
                for route in selected_routes:
                    route_set = set(route)
                    if not route_set.intersection(visited):
                        unique_routes.append(route)
                        visited.update(route_set)

                routes = unique_routes
            else:
                routes = []

        return RoutingIndividual(routes=routes, age=0)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, individual: RoutingIndividual) -> RoutingIndividual:
        """
        Apply destroy-repair mutation.

        Args:
            individual: Individual to mutate.

        Returns:
            Mutated individual.
        """
        if not individual.routes:
            return individual.copy()

        mutant = individual.copy()

        try:
            # Destroy: Remove random nodes
            n_remove = max(1, self.params.n_removal)
            partial_routes, removed_nodes = random_removal(mutant.routes, n_remove, rng=self.rng)

            # Repair: Reinsert removed nodes
            if removed_nodes:
                mutant.routes = greedy_insertion(
                    partial_routes,
                    removed_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    R=self.R,
                    mandatory_nodes=self.mandatory_nodes,
                )

                # Apply local search refinement
                from logic.src.policies.other.local_search.local_search_aco import ACOLocalSearch

                ls = ACOLocalSearch(
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    self.params,
                )
                mutant.routes = ls.optimize(mutant.routes)

        except Exception:
            # If mutation fails, return copy
            pass

        return mutant

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Evaluate solution fitness (net profit).

        Fitness = Revenue - Cost × C
        Revenue = Σ(waste_collected × R)
        Cost = Total distance traveled

        Args:
            routes: Routing solution.

        Returns:
            Net profit (higher is better).
        """
        self.n_evaluations += 1

        if not routes:
            return 0.0

        revenue = sum(self.wastes.get(n, 0.0) * self.R for route in routes for n in route)
        cost = self._cost(routes)

        return revenue - cost * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing distance.

        Args:
            routes: List of routes.

        Returns:
            Total distance traveled.
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

    def _update_best(self, population: List[RoutingIndividual]):
        """
        Update global best solution.

        Args:
            population: Current population.
        """
        current_best = max(population, key=lambda ind: ind.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.copy()
