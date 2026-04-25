"""Parameter dataclasses for Constraint Programming (CP-SAT) solver.

Attributes:
    CPSATParams (class): Standardized parameters for the CP-SAT solver.

Example:
    >>> params = CPSATParams(num_days=5, time_limit=60.0)
    >>> config_dict = params.to_dict()
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CPSATParams:
    """Standardized parameters for the CP-SAT solver.

    Attributes:
        num_days (int): Number of days in the planning horizon.
        time_limit (float): Maximum time in seconds for the solver.
        search_workers (int): Number of parallel search workers.
        mip_gap (float): Relative gap for optimality tolerance.
        scaling_factor (int): Multiplier for floating-to-integer conversion.
        waste_weight (float): Multiplier for waste revenue in objective.
        cost_weight (float): Multiplier for travel cost in objective.
        overflow_penalty (float): Multiplier for overflow penalties in objective.
        mean_increment (float): Average daily waste generation rate.
        seed (Optional[int]): Random seed for reproducibility.
    """

    num_days: int = 3
    time_limit: float = 300.0
    search_workers: int = 8
    mip_gap: float = 0.01
    scaling_factor: int = 1000

    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 10.0
    mean_increment: float = 0.2

    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CPSATParams":
        """
        Create a CPSATParams instance from a raw configuration dictionary.

        Args:
            config: Dictionary containing parameter overrides.

        Returns:
            A CPSATParams instance with values mapped from the config.
        """
        return cls(
            num_days=int(config.get("num_days", 3)),
            time_limit=float(config.get("time_limit", 300.0)),
            search_workers=int(config.get("search_workers", 8)),
            mip_gap=float(config.get("mip_gap", 0.01)),
            scaling_factor=int(config.get("scaling_factor", 1000)),
            waste_weight=float(config.get("waste_weight", 1.0)),
            cost_weight=float(config.get("cost_weight", 1.0)),
            overflow_penalty=float(config.get("overflow_penalty", 10.0)),
            mean_increment=float(config.get("mean_increment", 0.2)),
            seed=config.get("seed"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the parameters.
        """
        return {
            "num_days": self.num_days,
            "time_limit": self.time_limit,
            "search_workers": self.search_workers,
            "mip_gap": self.mip_gap,
            "scaling_factor": self.scaling_factor,
            "waste_weight": self.waste_weight,
            "cost_weight": self.cost_weight,
            "overflow_penalty": self.overflow_penalty,
            "mean_increment": self.mean_increment,
            "seed": self.seed,
        }
