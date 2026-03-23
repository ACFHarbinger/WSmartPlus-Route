"""
ACO Parameters Module.

This module defines the configuration parameters for the K-Sparse Ant Colony
Optimization algorithm. It uses a dataclass to store and validate hyperparameters.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization_k_sparse.params import KSACOParams
    >>> params = KSACOParams(n_ants=20, rho=0.1)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from logic.src.configs.policies import KSparseACOConfig


@dataclass
class KSACOParams:
    """
    Parameters for K-Sparse Ant Colony Optimization.

    Attributes:
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
        local_search_iterations: Number of iterations for local search.
        elitist_weight: Weight for best-so-far solution in pheromone update.
    """

    n_ants: int = 20
    k_sparse: int = 15
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    q0: float = 0.9
    tau_0: Optional[float] = None
    tau_min: float = 0.001
    tau_max: float = 10.0
    max_iterations: int = 100
    time_limit: float = 30.0
    local_search: bool = True
    local_search_iterations: int = 100
    elitist_weight: float = 1.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not (0 < self.rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {self.rho}")
        if not (0 <= self.q0 <= 1):
            raise ValueError(f"Exploitation probability q0 must be in [0, 1], got {self.q0}")

    @classmethod
    def from_config(cls, config: Union[KSparseACOConfig, Dict[str, Any]]) -> KSACOParams:
        """Create KSACOParams from an KSparseACOConfig dataclass or dict.

        Args:
            config: KSparseACOConfig dataclass or dict with solver parameters.

        Returns:
            KSACOParams instance with values from config.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            n_ants=getattr(config, "n_ants", 20),
            k_sparse=getattr(config, "k_sparse", 15),
            alpha=getattr(config, "alpha", 1.0),
            beta=getattr(config, "beta", 2.0),
            rho=getattr(config, "rho", 0.1),
            q0=getattr(config, "q0", 0.9),
            tau_0=getattr(config, "tau_0", None),
            tau_min=getattr(config, "tau_min", 0.001),
            tau_max=getattr(config, "tau_max", 10.0),
            max_iterations=getattr(config, "max_iterations", 100),
            time_limit=getattr(config, "time_limit", 30.0),
            local_search=getattr(config, "local_search", True),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            elitist_weight=getattr(config, "elitist_weight", 1.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
