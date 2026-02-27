"""
GA (Genetic Algorithm) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GAConfig:
    """Configuration for the Genetic Algorithm policy."""

    engine: str = "ga"
    pop_size: int = 30
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    n_removal: int = 2
    time_limit: float = 60.0
    vrpp: bool = True
    must_go: Optional[List[Any]] = field(default_factory=list)
    post_processing: Optional[List[Any]] = field(default_factory=list)
