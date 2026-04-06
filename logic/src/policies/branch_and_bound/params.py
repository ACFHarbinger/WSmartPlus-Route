"""
Parameter dataclasses for Branch-and-Bound solvers.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BBParams:
    """
    Standardized parameters for Branch-and-Bound solvers (MTZ and DFJ).
    """

    # Core Solver Parameters
    time_limit: float = 60.0
    mip_gap: float = 0.01
    seed: int = 42

    # MTZ Specific Parameters
    branching_strategy: str = "strong"  # "strong", "most_fractional", "least_fractional"
    strong_branching_limit: int = 5

    # Formulation
    formulation: str = "dfj"  # "dfj" or "mtz"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BBParams":
        """
        Create a BBParams instance from a raw configuration dictionary.

        Args:
            config: Dictionary containing parameter overrides.

        Returns:
            A BBParams instance with values mapped from the config.
        """
        return cls(
            time_limit=float(config.get("time_limit", 60.0)),
            mip_gap=float(config.get("mip_gap", 0.01)),
            seed=int(config.get("seed", 42)),
            branching_strategy=str(config.get("branching_strategy", "strong")),
            strong_branching_limit=int(config.get("strong_branching_limit", 5)),
            formulation=str(config.get("formulation", "dfj")),
        )
