"""
Configuration parameters for the Adaptive Large Neighborhood Search (ALNS).
"""


class ALNSParams:
    """
    Configuration parameters for the ALNS solver.

    Attributes:
        time_limit (int): Maximum runtime in seconds. Default: 10
        max_iterations (int): Maximum number of ALNS iterations. Default: 1000
        start_temp (float): Initial temperature for simulated annealing. Default: 100
        cooling_rate (float): Temperature decay factor per iteration. Default: 0.995
        reaction_factor (float): Learning rate for operator weight updates (rho). Default: 0.1
        min_removal (int): Minimum number of nodes to remove. Default: 1
        max_removal_pct (float): Maximum percentage of nodes to remove. Default: 0.3
    """

    def __init__(
        self,
        time_limit: int = 10,
        max_iterations: int = 1000,
        start_temp: float = 100,
        cooling_rate: float = 0.995,
        reaction_factor: float = 0.1,
        min_removal: int = 1,
        max_removal_pct: float = 0.3,
    ):
        """
        Initialize ALNS parameters.

        Args:
            time_limit (int): Max runtime in seconds.
            max_iterations (int): Max iterations.
            start_temp (float): Initial temperature.
            cooling_rate (float): Cooling rate.
            reaction_factor (float): Reaction factor.
            min_removal (int): Min nodes to remove.
            max_removal_pct (float): Max percentage of nodes to remove.
        """
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.start_temp = start_temp
        self.cooling_rate = cooling_rate
        self.reaction_factor = reaction_factor  # rho
        self.min_removal = min_removal
        self.max_removal_pct = max_removal_pct
