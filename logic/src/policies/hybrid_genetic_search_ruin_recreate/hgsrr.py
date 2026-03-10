"""
Hybrid Genetic Search with Ruin-and-Recreate (HGSRR) implementation.

Combines evolutionary search with adaptive Large Neighborhood Search operators
for solving the Capacitated Vehicle Routing Problem with Profits.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.hybrid_genetic_search import Individual
from logic.src.policies.hybrid_genetic_search.evolution import evaluate, update_biased_fitness
from logic.src.policies.hybrid_genetic_search.split import LinearSplit
from logic.src.policies.other.operators.crossover import ordered_crossover
from logic.src.tracking.viz_mixin import PolicyVizMixin

from .params import HGSRRParams
from .ruin_recreate import AdaptiveOperatorManager, RuinRecreateOperator


class HGSRRSolver(PolicyVizMixin):
    """
    Hybrid Genetic Search with Ruin-and-Recreate for VRPP.

    This algorithm extends standard HGS by replacing traditional local search
    with adaptive destroy/repair operators (ruin-and-recreate).
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HGSRRParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the HGSRR solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: HGSRR parameters.
            mandatory_nodes: List of local node indices that MUST be visited.
            seed: Random seed for reproducibility.
        """
        self.d = dist_matrix
        self.wastes = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(seed) if seed is not None else random.Random()

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Split manager for evaluating giant tours
        self.split_manager = LinearSplit(dist_matrix, wastes, capacity, R, C, params.max_vehicles, mandatory_nodes)

        # Ruin-and-recreate operator with adaptive selection
        self.rr_operator = RuinRecreateOperator(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            revenue=R,
            cost_unit=C,
            params=params,
            split_manager=self.split_manager,
            seed=seed,
        )

        # Adaptive operator manager
        self.operator_manager = AdaptiveOperatorManager(
            destroy_operators=params.destroy_operators,
            repair_operators=params.repair_operators,
            reaction_factor=params.reaction_factor,
            decay_parameter=params.decay_parameter,
        )

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the HGSRR algorithm.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        population: List[Individual] = []

        # 1. Initial Population
        for _ in range(self.params.population_size):
            gt = self.nodes[:]
            self.random.shuffle(gt)
            ind = Individual(gt)
            evaluate(ind, self.split_manager)
            population.append(ind)

        current_alpha = self.params.alpha_diversity
        update_biased_fitness(population, self.params.elite_size, current_alpha, self.params.neighbor_list_size)

        start_time = time.process_time()
        it = 0
        last_improvement_it = 0
        best_profit_so_far = max(ind.profit_score for ind in population)
        best_cost_so_far = min(ind.cost for ind in population if ind.profit_score == best_profit_so_far)

        while it < self.params.n_generations:
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break
            it += 1

            # 2. Selection & Crossover
            p1, p2 = self._select_parents(population)
            # Skip crossover if giant tours are too small (need at least 2 nodes)
            if len(p1.giant_tour) >= 2 and len(p2.giant_tour) >= 2:
                child = ordered_crossover(p1, p2, rng=self.random)
            else:
                # If tours are too small, just copy one parent
                child = Individual(p1.giant_tour[:])

            # 3. Adaptive Ruin-and-Recreate (Mutation)
            if self.random.random() < self.params.mutation_rate:
                # Select operators adaptively
                destroy_op, repair_op = self.operator_manager.select_operators(self.random)

                # Apply ruin-and-recreate
                child_improved = self.rr_operator.apply(
                    individual=child,
                    destroy_operator=destroy_op,
                    repair_operator=repair_op,
                )

                # Score the operators based on improvement
                score = self._compute_operator_score(child_improved, best_profit_so_far, best_cost_so_far, p1, p2)
                self.operator_manager.update_scores(destroy_op, repair_op, score)

                child = child_improved

            # Re-evaluate child
            evaluate(child, self.split_manager)
            population.append(child)

            # Track improvements
            if child.profit_score > best_profit_so_far:
                best_profit_so_far = child.profit_score
                best_cost_so_far = child.cost
                last_improvement_it = it
            elif child.profit_score == best_profit_so_far and child.cost < best_cost_so_far:
                best_cost_so_far = child.cost
                last_improvement_it = it

            # Adaptive alpha diversity
            avg_dist = np.mean([ind.dist_to_parents for ind in population])
            if avg_dist < self.params.min_diversity:
                current_alpha = min(1.0, current_alpha + self.params.diversity_change_rate)
            elif it - last_improvement_it > self.params.no_improvement_threshold:
                current_alpha = max(0.0, current_alpha - self.params.diversity_change_rate)

            # 4. Survivor Selection
            if len(population) > self.params.population_size * self.params.survivor_threshold:
                update_biased_fitness(population, self.params.elite_size, current_alpha, self.params.neighbor_list_size)
                population.sort(key=lambda x: x.fitness)
                population = population[: self.params.population_size]

            # Decay operator weights periodically
            if it % 10 == 0:
                self.operator_manager.decay_weights()

            self._viz_record(
                iteration=it,
                best_profit=best_profit_so_far,
                child_profit=child.profit_score,
                child_cost=child.cost,
                population_size=len(population),
                operator_entropy=self.operator_manager.entropy(),
            )

        # Final evaluation and selection
        update_biased_fitness(population, self.params.elite_size, current_alpha, self.params.neighbor_list_size)
        best_ind = min(population, key=lambda x: -x.profit_score)

        return best_ind.routes, best_ind.profit_score, best_ind.cost

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Select two parents using binary tournament selection."""

        def tournament():
            """Perform a binary tournament selection."""
            i1, i2 = self.random.sample(population, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()

    def _compute_operator_score(
        self,
        child: Individual,
        best_profit: float,
        best_cost: float,
        p1: Individual,
        p2: Individual,
    ) -> float:
        """
        Compute adaptive operator score based on solution quality improvement.

        Args:
            child: The offspring individual.
            best_profit: Current best profit in population.
            best_cost: Current best cost for the best profit.
            p1: First parent.
            p2: Second parent.

        Returns:
            Score value (sigma_1, sigma_2, sigma_3, or 0).
        """
        # New global best
        if child.profit_score > best_profit or (child.profit_score == best_profit and child.cost < best_cost):
            return self.params.score_sigma_1

        # Better than either parent
        parent_best_profit = max(p1.profit_score, p2.profit_score)
        if child.profit_score > parent_best_profit:
            return self.params.score_sigma_2

        # Accepted (added to population)
        return self.params.score_sigma_3


def run_hgsrr(dist_matrix, wastes, capacity, R, C, values, mandatory_nodes=None, *args):
    """
    Main HGSRR entry point.

    Args:
        dist_matrix: Distance matrix.
        wastes: Bin wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Dictionary of parameters and config.
        mandatory_nodes: List of local node indices that MUST be visited.
        *args: Additional arguments (ignored).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    if len(dist_matrix) <= 1:
        return [], 0.0, 0.0

    if len(dist_matrix) == 2:
        d = wastes.get(1, 0)
        if d <= capacity:
            cost = dist_matrix[0][1] + dist_matrix[1][0]
            profit = d * R
            return [[1]], profit, C * cost
        else:
            return [], 0.0, 0.0

    params = HGSRRParams(
        time_limit=values.get("time_limit", 10),
        population_size=values.get("population_size", 50),
        elite_size=values.get("elite_size", 10),
        mutation_rate=values.get("mutation_rate", 0.3),
        n_generations=values.get("n_generations", 100),
        alpha_diversity=values.get("alpha_diversity", 0.5),
        min_diversity=values.get("min_diversity", 0.2),
        diversity_change_rate=values.get("diversity_change_rate", 0.05),
        no_improvement_threshold=values.get("no_improvement_threshold", 20),
        survivor_threshold=values.get("survivor_threshold", 2.0),
        max_vehicles=values.get("max_vehicles", 0),
        crossover_rate=values.get("crossover_rate", 0.7),
        neighbor_list_size=values.get("neighbor_list_size", 10),
        # Ruin-recreate specific
        min_removal_pct=values.get("min_removal_pct", 0.1),
        max_removal_pct=values.get("max_removal_pct", 0.4),
        noise_factor=values.get("noise_factor", 0.015),
        reaction_factor=values.get("reaction_factor", 0.1),
        decay_parameter=values.get("decay_parameter", 0.95),
        seed=values.get("seed"),
    )

    solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes, seed=values.get("seed"))
    return solver.solve()
