"""
Configuration parameters for the Genetic Programming Hyper-Heuristic (GPHH) solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class GPHHParams:
    """
    Configuration parameters for the GPHH solver.

    GPHH evolves selection policies (GP trees) that govern which Low-Level
    Heuristic (LLH) to apply at each step, rather than evolving routing
    solutions directly.  The best policy tree is then applied for a longer
    run to produce the final solution.

    LLH Pool (5 operators, indices 0-4):
      L0: random_removal  + greedy_insertion
      L1: worst_removal   + regret_2_insertion
      L2: cluster_removal + greedy_insertion
      L3: worst_removal   + greedy_insertion
      L4: random_removal  + regret_2_insertion

    GP Trees are binary expression trees of depth ≤ `tree_depth`.
    Terminal nodes yield feature scalars; function nodes combine them.

    Attributes:
        gp_pop_size: Number of GP trees in the meta-population.
        max_gp_generations: GP evolution generations.
        eval_steps: LLH applications per tree fitness evaluation.
        apply_steps: LLH applications when running the best tree at the end.
        tree_depth: Maximum depth of GP expression trees.
        tournament_size: Tournament size for GP selection.
        n_llh: Number of Low-Level Heuristics in the pool.
        n_removal: Nodes removed per LLH destroy step.
        time_limit: Wall-clock time limit in seconds.
        parsimony_coefficient: Weight for tree size penalty (0.0 = no penalty).
    """

    gp_pop_size: int = 20
    max_gp_generations: int = 30
    eval_steps: int = 50
    apply_steps: int = 200
    tree_depth: int = 3
    tournament_size: int = 3
    n_llh: int = 5
    n_removal: int = 2
    time_limit: float = 60.0
    parsimony_coefficient: float = 0.0

    # Infrastructure
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "GPHHParams":
        """Create parameters from a configuration object."""
        return cls(
            gp_pop_size=getattr(config, "gp_pop_size", 20),
            max_gp_generations=getattr(config, "max_gp_generations", 30),
            eval_steps=getattr(config, "eval_steps", 50),
            apply_steps=getattr(config, "apply_steps", 200),
            tree_depth=getattr(config, "tree_depth", 3),
            tournament_size=getattr(config, "tournament_size", 3),
            n_llh=getattr(config, "n_llh", 5),
            n_removal=getattr(config, "n_removal", 2),
            time_limit=getattr(config, "time_limit", 60.0),
            parsimony_coefficient=getattr(config, "parsimony_coefficient", 0.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
