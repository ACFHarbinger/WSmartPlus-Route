"""
Ensemble Move Acceptance (EMA) parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EMAParams:
    """
    Configuration for the Ensemble Move Acceptance (EMA) solver.

    Rule choices: "G-AND", "G-OR", "G-VOT", "G-PVO"

    Attributes:
        max_iterations: Total LLH applications.
        rule: Method to combine sub-criteria decisions.
        criteria: List of criteria names to include (e.g., ["sa", "gd", "ie"]).
        sub_params: Dictionary mapping criteria name to its parameters.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
        time_limit: Wall-clock time limit in seconds.
    """

    max_iterations: int = 1000
    rule: str = "G-VOT"
    criteria: List[str] = field(default_factory=lambda: ["sa", "gd", "ie"])
    sub_params: Dict[str, Any] = field(default_factory=dict)
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
