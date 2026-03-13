"""
Stochastic Tournament Genetic Algorithm (STGA) for VRPP.

This is the rigorous implementation of the League Championship Algorithm (LCA)
with proper Operations Research terminology, replacing sports metaphors with
canonical metaheuristic concepts:

TERMINOLOGY MAPPING (LCA → MA-TB):
- "Teams" → Population (candidate solutions)
- "League Schedule" → Round-Robin Pairwise Matching
- "Playing Strength" → Fitness (objective function value)
- "Match Outcome" → Pairwise Comparison with Infeasibility Tolerance
- "Team Formation" → Crossover/Mutation Operators
- "Seasons" → Generations

Algorithm Structure (Kashan, 2013):
    1. Initialize population of N solutions
    2. For each generation:
        a. Round-Robin Schedule: Random pairwise matching
        b. For each match (i vs j):
            - Determine winner using infeasibility tolerance
            - If |fitness_i - fitness_j| ≤ tolerance: random winner (diversity)
            - Otherwise: higher fitness wins
            - Loser generates new solution (crossover or mutation)
            - Loser ALWAYS accepts new solution (no elitism within match)
        c. Update global best

Key Feature: Infeasibility Tolerance
    - Allows solutions with similar fitness to compete randomly
    - Preserves diversity and prevents premature convergence
    - Critical for bridging isolated feasible basins in constrained problems

Reference:
    Kashan, A. H. (2013). "League Championship Algorithm (LCA): An algorithm
    for global optimization inspired by sport championships."
    Applied Soft Computing, 13(5), 2171-2200.

IMPORTANT: This implementation EXACTLY matches the LCA algorithm with only
           terminology changed from sports metaphors to OR terminology.
"""

import contextlib
import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.other.local_search.local_search_aco import ACOLocalSearch
from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..other.operators import greedy_insertion, worst_removal
from .params import MemeticAlgorithmToleranceBasedParams


class MemeticAlgorithmToleranceBasedSolver(PolicyVizMixin):
    """
    Memetic Algorithm with Tolerance-Based Selection (MA-TB) for VRPP.

    Uses round-robin pairwise matching with infeasibility tolerance for
    diversity-preserving competition-based evolution.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MemeticAlgorithmToleranceBasedParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Memetic Algorithm Tolerance Based solver.

        Args:
            dist_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 = depot.
            wastes: Dictionary mapping node index to waste/profit value.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit of waste collected.
            C: Cost per unit of distance traveled.
            params: MA-TB configuration parameters.
            mandatory_nodes: List of nodes that must be visited.
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
        self.mandatory_set = set(self.mandatory_nodes)
        self.random = random.Random(seed) if seed is not None else random.Random()

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute Memetic Algorithm with Tolerance-Based Selection.

        Returns:
            Tuple of (routes, profit, cost):
                - routes: Best routing solution found
                - profit: Net profit of the solution
                - cost: Total routing distance
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # Phase 1: Population Initialization
        population: List[List[List[int]]] = [self._build_solution() for _ in range(self.params.population_size)]
        fitness_values: List[float] = [self._evaluate(solution) for solution in population]

        # Track global best
        best_idx = int(np.argmax(fitness_values))
        best_solution = copy.deepcopy(population[best_idx])
        best_profit = fitness_values[best_idx]
        best_cost = self._cost(best_solution)

        # Main Evolution Loop
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Phase 2: Round-Robin Pairwise Matching
            # Create random schedule for this generation
            match_order = list(range(self.params.population_size))
            self.random.shuffle(match_order)

            # Execute pairwise matches
            for k in range(0, len(match_order) - 1, 2):
                solution_a_idx = match_order[k]
                solution_b_idx = match_order[k + 1]

                fitness_a = fitness_values[solution_a_idx]
                fitness_b = fitness_values[solution_b_idx]

                # Phase 3: Winner Determination with Infeasibility Tolerance
                winner_idx, loser_idx = self._determine_winner(solution_a_idx, solution_b_idx, fitness_a, fitness_b)

                # Phase 4: Loser Generates New Solution
                if self.random.random() < self.params.recombination_rate:
                    # Recombination: Loser learns from winner
                    new_solution = self._recombine(population[loser_idx], population[winner_idx])
                else:
                    # Mutation: Loser perturbs itself
                    new_solution = self._mutate(population[loser_idx])

                new_fitness = self._evaluate(new_solution)

                # Phase 5: Update Loser (LCA always updates loser)
                population[loser_idx] = new_solution
                fitness_values[loser_idx] = new_fitness

                # Update global best
                if new_fitness > best_profit:
                    best_solution = copy.deepcopy(new_solution)
                    best_profit = new_fitness
                    best_cost = self._cost(best_solution)

            # Visualization tracking
            self._viz_record(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                population_size=self.params.population_size,
            )

        return best_solution, best_profit, best_cost

    # ------------------------------------------------------------------
    # Population Initialization
    # ------------------------------------------------------------------

    def _build_solution(self) -> List[List[int]]:
        """
        Initialize a single routing solution.

        Uses nearest-neighbor heuristic with randomized node ordering
        to create diverse initial population.

        Returns:
            Feasible routing solution as list of routes.

        Complexity: O(n²) for NN construction.
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
            rng=self.random,
        )
        return routes

    # ------------------------------------------------------------------
    # Competition Operators
    # ------------------------------------------------------------------

    def _determine_winner(
        self,
        solution_a_idx: int,
        solution_b_idx: int,
        fitness_a: float,
        fitness_b: float,
    ) -> Tuple[int, int]:
        """
        Determine match winner using infeasibility tolerance.

        This is the KEY FEATURE of LCA that distinguishes it from standard
        tournament selection. Solutions with similar fitness compete randomly,
        preserving diversity and allowing exploration of alternative basins.

        Rigorous Interpretation of LCA "Match Outcome":
            - Metaphor: "Teams within tolerance compete fairly"
            - Rigorous: "Solutions within ε-tolerance have stochastic selection"

        Mathematical Formulation:
            tolerance = tolerance_pct × (|fitness_a| + |fitness_b|) / 2
            if |fitness_a - fitness_b| ≤ tolerance:
                winner = random choice (preserves diversity)
            else:
                winner = argmax(fitness_a, fitness_b)

        Args:
            solution_a_idx: Index of first solution.
            solution_b_idx: Index of second solution.
            fitness_a: Fitness of first solution.
            fitness_b: Fitness of second solution.

        Returns:
            Tuple of (winner_idx, loser_idx).

        Complexity: O(1) constant time.
        """
        # Calculate fitness difference
        delta = abs(fitness_a - fitness_b)

        # Calculate tolerance threshold
        tolerance = self.params.tolerance_pct * (abs(fitness_a) + abs(fitness_b) + 1e-9) / 2.0

        if delta <= tolerance:
            # Close match - random winner (diversity preservation)
            if self.random.random() < 0.5:
                winner_idx, loser_idx = solution_a_idx, solution_b_idx
            else:
                winner_idx, loser_idx = solution_b_idx, solution_a_idx
        elif fitness_a >= fitness_b:
            # Clear winner based on fitness
            winner_idx, loser_idx = solution_a_idx, solution_b_idx
        else:
            winner_idx, loser_idx = solution_b_idx, solution_a_idx

        return winner_idx, loser_idx

    # ------------------------------------------------------------------
    # Evolution Operators
    # ------------------------------------------------------------------

    def _recombine(self, loser_solution: List[List[int]], winner_solution: List[List[int]]) -> List[List[int]]:
        """
        Generate new solution by recombining loser with winner.

        The loser adopts a contiguous segment of nodes from the winner's
        solution that it doesn't already visit. This implements the LCA
        "formation update" where losing teams learn from winners.

        Rigorous Interpretation of LCA "Formation Update":
            - Metaphor: "Losing team analyzes winner's strategy"
            - Rigorous: "Segment-based crossover with guided node injection"

        Algorithm:
            1. Extract flat node sequence from winner
            2. Select contiguous segment (~25% of winner's nodes)
            3. Identify nodes in segment not in loser
            4. Remove worst nodes from loser to make room
            5. Insert new nodes via greedy insertion
            6. Apply local search refinement

        Args:
            loser_solution: Losing solution's routes.
            winner_solution: Winning solution's routes.

        Returns:
            New routing solution (offspring).

        Complexity: O(n²) for removal + insertion + local search.
        """
        # Extract winner's node sequence
        winner_nodes = [node for route in winner_solution for node in route]
        loser_visited = {node for route in loser_solution for node in route}

        # Handle edge case: winner has too few nodes
        if len(winner_nodes) < 2:
            return self._mutate(loser_solution)

        # Select contiguous segment from winner (~25% of nodes)
        segment_length = max(1, len(winner_nodes) // 4)
        start_idx = self.random.randint(0, max(0, len(winner_nodes) - segment_length))
        segment = winner_nodes[start_idx : start_idx + segment_length]

        # Identify new nodes (in segment but not in loser)
        new_nodes = [node for node in segment if node not in loser_visited]

        # Start with copy of loser
        child_solution = copy.deepcopy(loser_solution)

        # Remove worst nodes to make room for new nodes
        n_remove = min(len(new_nodes), max(1, self.params.perturbation_strength))
        with contextlib.suppress(Exception):
            child_solution, _ = worst_removal(child_solution, n_remove, self.dist_matrix)

        # Insert new nodes from winner's segment
        if new_nodes:
            with contextlib.suppress(Exception):
                child_solution = greedy_insertion(
                    child_solution,
                    new_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    R=self.R,
                    mandatory_nodes=self.mandatory_nodes,
                )

        # Apply local search refinement
        ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
        return ls.optimize(child_solution)

    def _mutate(self, solution: List[List[int]]) -> List[List[int]]:
        """
        Apply mutation operator to solution.

        Uses destroy-repair strategy: remove worst nodes and reinsert greedily.
        This provides local exploration when recombination is not used.

        Rigorous Interpretation of LCA "Self-Improvement":
            - Metaphor: "Team improves formation independently"
            - Rigorous: "Local perturbation via worst-removal + greedy-repair"

        Args:
            solution: Current routing solution.

        Returns:
            Mutated routing solution.

        Complexity: O(n²) for removal + insertion + local search.
        """
        n_remove = max(3, self.params.perturbation_strength)
        try:
            # Destroy: Remove worst nodes
            partial_solution, removed_nodes = worst_removal(solution, n_remove, self.dist_matrix)

            # Repair: Reinsert greedily
            repaired_solution = greedy_insertion(
                partial_solution,
                removed_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )

            # Refinement: Local search
            ls = ACOLocalSearch(self.dist_matrix, self.wastes, self.capacity, self.R, self.C, self.params)
            return ls.optimize(repaired_solution)
        except Exception:
            # If mutation fails, return copy of original
            return copy.deepcopy(solution)

    # ------------------------------------------------------------------
    # Fitness Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Evaluate net profit of a routing solution.

        Fitness = Revenue - Cost × C
        Revenue = Σ(waste_collected × R)
        Cost = Total distance traveled

        Args:
            routes: List of vehicle routes.

        Returns:
            Net profit (higher is better).

        Complexity: O(n) for route traversal.
        """
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(node, 0.0) * self.R for route in routes for node in route)
        return revenue - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing distance.

        Includes depot→first_node, inter-node, and last_node→depot distances.

        Args:
            routes: List of vehicle routes.

        Returns:
            Total distance traveled.

        Complexity: O(n) for route traversal.
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
            # Last node back to depot
            total += self.dist_matrix[route[-1]][0]
        return total
