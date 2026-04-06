"""
Configuration parameters for the Branch-and-Price-and-Cut (BPC) solver.

Based on Barnhart et al. (1998) and standard VRPP practices.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class BPCParams:
    """
    Configuration parameters for the Branch-and-Price-and-Cut solver.

    Attributes:
        optimality_gap: Relative gap for proven optimality.
        early_termination_gap: Gap at which we stop the search.
        use_ng_routes: Whether to use ng-route relaxation in pricing.
        ng_neighborhood_size: Size of ng-neighborhoods for relaxation.
        enable_fractional_capacity_cuts: Whether to enable expensive fractional capacity cuts.
        enable_comb_cuts: Whether to enable heuristic comb inequalities.
    """

    time_limit: float = 60.0
    engine: str = "custom"
    profit_aware_operators: bool = False
    vrpp: bool = True
    seed: Optional[int] = None
    search_strategy: str = "depth_first"
    cutting_planes: str = "rcc"
    branching_strategy: str = "divergence"
    max_cg_iterations: int = 50
    max_cuts_per_iteration: int = 5
    max_routes_per_pricing: int = 5
    max_bb_nodes: int = 1000
    optimality_gap: float = 1e-4
    early_termination_gap: float = 1e-3
    use_ng_routes: bool = True
    ng_neighborhood_size: int = 8
    enable_fractional_capacity_cuts: bool = True
    enable_comb_cuts: bool = False

    @classmethod
    def from_config(cls, config: Any) -> BPCParams:
        """Create BPCParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            time_limit=getattr(config, "time_limit", 60.0),
            engine=getattr(config, "engine", "custom"),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            vrpp=getattr(config, "vrpp", True),
            seed=getattr(config, "seed", None),
            search_strategy=getattr(config, "search_strategy", "depth_first"),
            cutting_planes=getattr(config, "cutting_planes", "rcc"),
            branching_strategy=getattr(config, "branching_strategy", "divergence"),
            max_cg_iterations=getattr(config, "max_cg_iterations", 50),
            max_cuts_per_iteration=getattr(config, "max_cuts_per_iteration", 5),
            max_routes_per_pricing=getattr(config, "max_routes_per_pricing", 5),
            max_bb_nodes=getattr(config, "max_bb_nodes", 1000),
            optimality_gap=getattr(config, "optimality_gap", 1e-4),
            early_termination_gap=getattr(config, "early_termination_gap", 1e-3),
            use_ng_routes=getattr(config, "use_ng_routes", True),
            ng_neighborhood_size=getattr(config, "ng_neighborhood_size", 8),
            enable_fractional_capacity_cuts=getattr(config, "enable_fractional_capacity_cuts", True),
            enable_comb_cuts=getattr(config, "enable_comb_cuts", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert BPCParams to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
