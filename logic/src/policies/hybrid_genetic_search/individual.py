from typing import List

"""
HGS Individual Module.

This module defines the Individual class used in the Hybrid Genetic Search (HGS)
algorithm. It represents a single solution in the population, storing genotype
(giant tour) and phenotype (routes), along with cost and fitness metrics.

Attributes:
    None

Example:
    >>> from logic.src.policies.hybrid_genetic_search.individual import Individual
    >>> ind = Individual(genotype=[1, 5, 2, ...], cost=100.0)
"""


class Individual:
    """
    Individual solution representation for genetic algorithm.
    Follows Vidal et al. (2022) HGS-CVRP implementation.
    """

    def __init__(self, giant_tour: List[int], expand_pool: bool = False):
        """
        Initialize an individual with a giant tour.

        Args:
            giant_tour: A list representing the visit order of all clients.
            expand_pool: If True (VRPP mode), repair operators consider all unvisited nodes.
                        If False (CVRP mode), repair operators only consider removed nodes.
        """
        self.giant_tour = giant_tour
        self.routes: List[List[int]] = []
        self.fitness = -float("inf")
        self.profit_score = -float("inf")
        self.cost = 0.0
        self.revenue = 0.0

        # Feasibility tracking
        self.is_feasible = True
        self.capacity_violation = 0.0  # Total capacity excess across all routes
        self.penalized_cost = 0.0  # Cost including penalty for violations

        # Diversity and ranking
        self.dist_to_parents = 0.0
        self.rank_profit = 0
        self.rank_diversity = 0
        self.is_coached = False

        # VRPP-specific: controls whether repair operators expand node pool
        self.expand_pool = expand_pool

    def __lt__(self, other: "Individual") -> bool:
        """Comparison based on biased fitness (used for sorting)."""
        return self.fitness < other.fitness
