"""
SISR (Slack Induction by String Removal) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..other.must_go import MustGoConfig
from ..other.post_processing import PostProcessingConfig


@dataclass
class SISRConfig:
    """Configuration for Slack Induction by String Removal (SISR) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum number of SISR iterations.
        start_temp: Initial temperature for simulated annealing acceptance.
        cooling_rate: Rate at which temperature decreases each iteration.
        max_string_len: Maximum length of string to remove.
        avg_string_len: Average length of strings to remove.
        blink_rate: Probability of 'blinking' during insertion.
        destroy_ratio: Fraction of solution to destroy per iteration.
        engine: Solver engine to use.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    time_limit: float = 10.0
    max_iterations: int = 1000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    max_string_len: int = 10
    avg_string_len: float = 3.0
    blink_rate: float = 0.01
    destroy_ratio: float = 0.2
    engine: str = "custom"
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
