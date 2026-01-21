"""
Core types and parameters for Hybrid Genetic Search (HGS).
"""


class Individual:
    """
    Individual solution representation for genetic algorithm.
    """

    def __init__(self, giant_tour: list[int]):
        """
        Initialize an individual with a giant tour.

        Args:
            giant_tour: A list representing the visit order of all clients.
        """
        self.giant_tour = giant_tour
        self.routes: list[list[int]] = []
        self.fitness = -float("inf")
        self.profit_score = -float("inf")
        self.cost = 0.0
        self.revenue = 0.0

        self.dist_to_parents = 0.0
        self.rank_profit = 0
        self.rank_diversity = 0

    def __lt__(self, other: "Individual") -> bool:
        """Comparison based on biased fitness (used for sorting)."""
        return self.fitness < other.fitness


class HGSParams:
    """
    Configuration parameters for Hybrid Genetic Search.
    """

    def __init__(
        self,
        time_limit: int = 10,
        population_size: int = 50,
        elite_size: int = 10,
        mutation_rate: float = 0.2,
        max_vehicles: int = 0,
    ):
        """
        Initialize HGS parameters.

        Args:
            time_limit: Maximum search time in seconds.
            population_size: Target population size.
            elite_size: Number of elite individuals for survivor selection.
            mutation_rate: Probability of applying local search improvement.
            max_vehicles: Maximum number of vehicles allowed.
        """
        self.time_limit = time_limit
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.max_vehicles = max_vehicles
