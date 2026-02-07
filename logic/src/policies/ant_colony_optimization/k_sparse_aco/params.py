"""
ACO Parameters module.

Defines hyperparameters for the K-Sparse Ant Colony Optimization algorithm.
"""

from typing import Optional


class ACOParams:
    """
    Parameters for K-Sparse Ant Colony Optimization.
    """

    def __init__(
        self,
        n_ants: int = 10,
        k_sparse: int = 15,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q0: float = 0.9,
        tau_0: Optional[float] = None,
        tau_min: float = 0.001,
        tau_max: float = 10.0,
        max_iterations: int = 100,
        time_limit: float = 30.0,
        local_search: bool = True,
        elitist_weight: float = 1.0,
    ):
        """
        Initialize ACO parameters.

        Args:
            n_ants: Number of ants per iteration.
            k_sparse: Number of pheromone values to retain per node (candidate list size).
            alpha: Pheromone importance exponent.
            beta: Heuristic (distance) importance exponent.
            rho: Pheromone evaporation rate (0 < rho < 1).
            q0: Exploitation probability for pseudo-random proportional rule.
            tau_0: Initial pheromone level.
            tau_min: Minimum pheromone level (for MMAS bounds).
            tau_max: Maximum pheromone level (for MMAS bounds).
            max_iterations: Maximum number of iterations.
            time_limit: Maximum runtime in seconds.
            local_search: Whether to apply local search (2-opt) to solutions.
            elitist_weight: Weight for best-so-far solution in pheromone update.
        """
        if not (0 < rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {rho}")
        if not (0 <= q0 <= 1):
            raise ValueError(f"Exploitation probability q0 must be in [0, 1], got {q0}")

        self.n_ants = n_ants
        self.k_sparse = k_sparse
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.local_search = local_search
        self.elitist_weight = elitist_weight
