r"""Parameter dataclasses for Logic-Based Benders Decomposition (LBBD) solver.

Attributes:
    LBBDParams: Dataclass for LBBD solver configuration.

Example:
    >>> params = LBBDParams(max_iterations=50)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LBBDParams:
    r"""Standardized parameters for the LBBD solver.

    Attributes:
        num_days (int): Planning horizon.
        stochastic_master (bool): Use stochastic master problem.
        mean_increment (float): Average waste increment per day.
        num_scenarios (int): Number of scenarios for evaluation.
        max_iterations (int): Maximum Benders iterations.
        benders_gap (float): Optimality gap for convergence.
        time_limit (float): Overall time limit in seconds.
        subproblem_timeout (float): Subproblem solver timeout.
        mip_gap (float): Gurobi MIP gap for the master problem.
        waste_weight (float): Objective weight for waste collected.
        cost_weight (float): Objective weight for travel cost.
        overflow_penalty (float): Penalty for bin overflow.
        use_nogood_cuts (bool): Enable logic-based no-good cuts.
        use_optimality_cuts (bool): Enable standard optimality cuts.
        seed (Optional[int]): Random seed for scenario generation.
    """

    num_days: int = 3
    stochastic_master: bool = False
    mean_increment: float = 0.2
    num_scenarios: int = 5
    max_iterations: int = 20
    benders_gap: float = 0.01
    time_limit: float = 300.0
    subproblem_timeout: float = 20.0
    mip_gap: float = 0.05
    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 10.0
    use_nogood_cuts: bool = True
    use_optimality_cuts: bool = True
    seed: Optional[int] = 42

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LBBDParams":
        """Create a LBBDParams instance from a raw configuration dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing parameter overrides.

        Returns:
            LBBDParams: A LBBDParams instance with values mapped from the config.
        """
        return cls(
            num_days=int(config.get("num_days", 3)),
            stochastic_master=bool(config.get("stochastic_master", False)),
            mean_increment=float(config.get("mean_increment", 0.2)),
            num_scenarios=int(config.get("num_scenarios", 5)),
            max_iterations=int(config.get("max_iterations", 20)),
            benders_gap=float(config.get("benders_gap", 0.01)),
            time_limit=float(config.get("time_limit", 300.0)),
            subproblem_timeout=float(config.get("subproblem_timeout", 20.0)),
            mip_gap=float(config.get("mip_gap", 0.05)),
            waste_weight=float(config.get("waste_weight", 1.0)),
            cost_weight=float(config.get("cost_weight", 1.0)),
            overflow_penalty=float(config.get("overflow_penalty", 10.0)),
            use_nogood_cuts=bool(config.get("use_nogood_cuts", True)),
            use_optimality_cuts=bool(config.get("use_optimality_cuts", True)),
            seed=config.get("seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of parameter values.
        """
        return {
            "num_days": self.num_days,
            "stochastic_master": self.stochastic_master,
            "mean_increment": self.mean_increment,
            "num_scenarios": self.num_scenarios,
            "max_iterations": self.max_iterations,
            "benders_gap": self.benders_gap,
            "time_limit": self.time_limit,
            "subproblem_timeout": self.subproblem_timeout,
            "mip_gap": self.mip_gap,
            "waste_weight": self.waste_weight,
            "cost_weight": self.cost_weight,
            "overflow_penalty": self.overflow_penalty,
            "use_nogood_cuts": self.use_nogood_cuts,
            "use_optimality_cuts": self.use_optimality_cuts,
            "seed": self.seed,
        }
