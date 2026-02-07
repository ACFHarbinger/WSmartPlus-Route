"""
ACO Parameters module for Hyper-Heuristic ACO.

Defines hyperparameters for the Hyper-Heuristic Ant Colony Optimization algorithm.
"""

from typing import List, Optional

from .hyper_operators import OPERATOR_NAMES


class HyperACOParams:
    """
    Parameters for Hyper-Heuristic ACO.
    """

    def __init__(
        self,
        n_ants: int = 10,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        tau_0: float = 1.0,
        tau_min: float = 0.01,
        tau_max: float = 10.0,
        max_iterations: int = 50,
        time_limit: float = 30.0,
        sequence_length: int = 5,
        q0: float = 0.9,
        operators: Optional[List[str]] = None,
    ):
        """
        Initialize Hyper-ACO parameters.

        Args:
            n_ants: Number of ants per iteration.
            alpha: Pheromone importance exponent.
            beta: Heuristic importance exponent.
            rho: Pheromone evaporation rate.
            tau_0: Initial pheromone level.
            tau_min: Minimum pheromone level (MMAS bounds).
            tau_max: Maximum pheromone level (MMAS bounds).
            max_iterations: Maximum number of iterations.
            time_limit: Maximum runtime in seconds.
            sequence_length: Length of operator sequence each ant constructs.
            q0: Exploitation probability for pseudo-random proportional rule.
            operators: List of operator names to include in the sequence construction.
        """
        if not (0 < rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {rho}")
        if not (0 <= q0 <= 1):
            raise ValueError(f"Exploitation probability q0 must be in [0, 1], got {q0}")

        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.sequence_length = sequence_length
        self.q0 = q0
        self.operators = operators if operators is not None else OPERATOR_NAMES.copy()
