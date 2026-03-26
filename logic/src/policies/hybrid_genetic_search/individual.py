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
    Follows Vidal et al. (2022) HGS-CVRP implementation with VRPP adaptations.

    Genotype/Phenotype Mapping:
        - Genotype: giant_tour (ordered list of all customer nodes, including unvisited)
        - Phenotype: routes (actual vehicle routes after Split algorithm)

    For VRPP (Vehicle Routing Problem with Profits):
        The Split algorithm may skip unprofitable nodes, creating a mismatch between
        genotype and phenotype. To maintain genetic consistency for crossover:

        1. giant_tour contains ALL nodes in a specific order (genotype)
        2. routes contains only visited nodes organized into vehicle routes (phenotype)
        3. Nodes in giant_tour but not in routes are implicitly "unvisited"
        4. Local search operators may move nodes between visited and unvisited sets
        5. Crossover operates on giant_tour, preserving genetic material for all nodes

    This approach (Option A: Implicit Dummy Route) treats unvisited nodes as being
    in a conceptual "dummy route" at the end of the giant tour.
    """

    def __init__(self, giant_tour: List[int], expand_pool: bool = False):
        """
        Initialize an individual with a giant tour.

        Args:
            giant_tour: A list representing the visit order of all clients (genotype).
                       For VRPP, this includes both visited and potentially unvisited nodes.
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

    def get_visited_nodes(self) -> set:
        """
        Get the set of nodes that are currently visited in the solution.

        Returns:
            Set of node IDs present in routes (phenotype).
        """
        return {node for route in self.routes for node in route}

    def get_unvisited_nodes(self) -> List[int]:
        """
        Get the list of nodes in the giant tour that are not currently visited.

        For VRPP, these are nodes that the Split algorithm decided to skip
        because they were unprofitable, or that local search removed.

        Returns:
            List of node IDs in giant_tour but not in routes, in giant_tour order.
        """
        visited = self.get_visited_nodes()
        return [node for node in self.giant_tour if node not in visited]

    def __lt__(self, other: "Individual") -> bool:
        """Comparison based on biased fitness (used for sorting)."""
        return self.fitness < other.fitness
