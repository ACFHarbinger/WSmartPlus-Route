"""
GPHH (Genetic Programming Hyper-Heuristic) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GPHHConfig:
    """
    Configuration for the Genetic Programming Hyper-Heuristic policy.

    Attributes:
        gp_pop_size: GP meta-population size (number of policy trees).
        max_gp_generations: GP evolution generations.
        eval_steps: LLH applications per tree fitness evaluation.
        apply_steps: LLH applications when running the best tree.
        tree_depth: Maximum depth of GP expression trees.
        tournament_size: Tournament size for GP parent selection.
        n_llh: Number of Low-Level Heuristics in the pool (fixed at 5).
        n_removal: Nodes removed per LLH destroy step.
        time_limit: Wall-clock time limit in seconds.
        vrpp: If True, solver operates in full VRPP mode.
        must_go: Must-go selection strategy config list.
        post_processing: Post-processing operation config list.
    """

    engine: str = "gphh"
    gp_pop_size: int = 20
    max_gp_generations: int = 30
    eval_steps: int = 50
    apply_steps: int = 200
    tree_depth: int = 3
    tournament_size: int = 3
    n_llh: int = 5
    n_removal: int = 2
    time_limit: float = 60.0
    vrpp: bool = True
    must_go: Optional[List[Any]] = field(default_factory=list)
    post_processing: Optional[List[Any]] = field(default_factory=list)
