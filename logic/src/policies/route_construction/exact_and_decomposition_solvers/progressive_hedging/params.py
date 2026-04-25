r"""Parameter dataclasses for Progressive Hedging (PH) solver.

Attributes:
    PHParams: Dataclass for PH solver configuration.

Example:
    >>> params = PHParams(rho=0.5, max_iterations=100)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PHParams:
    r"""Standardized parameters for the PH solver.

    Attributes:
        rho (float): Penalty parameter for consensus deviation.
        max_iterations (int): Maximum PH iterations.
        convergence_tol (float): Primal residual threshold for convergence.
        sub_solver (str): Solver to use for scenario subproblems.
        num_scenarios (int): Number of scenarios for evaluation.
        time_limit (float): Overall time limit in seconds.
        verbose (bool): Enable verbose logging.
        seed (Optional[int]): Random seed for scenario generation.
    """

    rho: float = 1.0
    max_iterations: int = 50
    convergence_tol: float = 0.01
    sub_solver: str = "bc"
    num_scenarios: int = 10
    time_limit: float = 300.0
    verbose: bool = True
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PHParams":
        """
        Create a PHParams instance from a raw configuration dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing parameter overrides.

        Returns:
            PHParams: A PHParams instance with values mapped from the config.
        """
        return cls(
            rho=float(config.get("rho", 1.0)),
            max_iterations=int(config.get("max_iterations", 50)),
            convergence_tol=float(config.get("convergence_tol", 0.01)),
            sub_solver=str(config.get("sub_solver", "bc")),
            num_scenarios=int(config.get("num_scenarios", 10)),
            time_limit=float(config.get("time_limit", 300.0)),
            verbose=bool(config.get("verbose", True)),
            seed=config.get("seed"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of parameter values.
        """
        return {
            "rho": self.rho,
            "max_iterations": self.max_iterations,
            "convergence_tol": self.convergence_tol,
            "sub_solver": self.sub_solver,
            "num_scenarios": self.num_scenarios,
            "time_limit": self.time_limit,
            "verbose": self.verbose,
            "seed": self.seed,
        }
