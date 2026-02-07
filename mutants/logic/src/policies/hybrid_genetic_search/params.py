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
