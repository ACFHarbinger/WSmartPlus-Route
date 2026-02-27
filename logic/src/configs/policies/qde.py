"""
QDE (Quantum-Inspired Differential Evolution) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class QDEConfig:
    """
    Configuration for the Quantum-Inspired Differential Evolution policy.

    Attributes:
        pop_size: Population size.
        F: Mutation scaling factor ∈ (0, 2].
        CR: Crossover probability ∈ [0, 1].
        max_iterations: Maximum DE generations.
        time_limit: Wall-clock time limit in seconds.
        n_removal: Nodes removed per collapse-repair step.
        vrpp: If True, solver operates in full VRPP mode (consider all bins).
        must_go: Must-go selection strategy config list.
        post_processing: Post-processing operation config list.
    """

    engine: str = "qde"
    pop_size: int = 20
    F: float = 0.5
    CR: float = 0.7
    max_iterations: int = 200
    time_limit: float = 60.0
    n_removal: int = 2
    vrpp: bool = True
    must_go: Optional[List[Any]] = field(default_factory=list)
    post_processing: Optional[List[Any]] = field(default_factory=list)
