"""
Configuration parameters for the Branch-and-Price-and-Cut (BPC) solver.

Based on Barnhart et al. (1998, 2000) and standard exact VRPP protocols.
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
        early_termination_gap: Gap at which we stop the search (heuristic shortcut).
        use_ng_routes: Whether to use ng-route relaxation in pricing.
        ng_neighborhood_size: Size of ng-neighborhoods for relaxation.
        enable_heuristic_rcc_separation: Whether to enable (heuristic) fractional RCC separation.
        enable_comb_cuts: Whether to enable heuristic comb inequalities.
        cut_orthogonality_threshold: cosine similarity ceiling for cut filtering.
        enable_column_pool_deduplication: Whether to enable column pool deduplication.
        enable_hybrid_search: Whether to enable hybrid DFS/BFS search.
        rc_tolerance: Minimum reduced cost to accept a new column.
        exact_mode: Whether to enable strict exact management (no dual smoothing).
        strong_branching_size: Number of candidates to evaluate in strong branching.


    Note:
        enable_heuristic_rcc_separation replaces the legacy enable_fractional_capacity_cuts
        to clarify that the connected-component-based separation is a heuristic,
        not an exact max-flow separation on the full support graph.
    """

    time_limit: float = 60.0
    profit_aware_operators: bool = False
    vrpp: bool = True
    seed: Optional[int] = None
    search_strategy: str = "depth_first"
    cutting_planes: str = "rcc"
    branching_strategy: str = "divergence"
    max_cg_iterations: int = 50
    max_cut_iterations: int = 5
    max_cuts_per_iteration: int = 5
    max_routes_per_pricing: int = 5
    max_bb_nodes: int = 1000
    optimality_gap: float = 1e-4
    early_termination_gap: float = 1e-3  # HEURISTIC SHORTCUT: See docstring for caveats.
    use_ng_routes: bool = True
    ng_neighborhood_size: int = 8
    enable_heuristic_rcc_separation: bool = True
    enable_comb_cuts: bool = False
    cut_orthogonality_threshold: float = 0.8
    use_spatial_partitioning: bool = False
    # Disabled by default. The current implementation is a heuristic proxy
    # that returns the top pre-sorted divergence candidate without solving
    # any child LP relaxations. When True, it bypasses fleet-size and
    # Ryan-Foster branching levels at every fractional node for no benefit.
    # Enable only after implementing full LP-based strong branching
    # (2×N child LP solves per candidate).
    enable_strong_branching_heuristic: bool = False
    enable_column_pool_deduplication: bool = True
    enable_hybrid_search: bool = False  # Task 13: DFS -> BFS switch
    rc_tolerance: float = 1e-5  # Fix 8: Minimum reduced cost to accept a new column
    exact_mode: bool = False  # Task 3: Enable strict exact management
    strong_branching_size: int = 5  # Task 1: Number of candidates for strong branching

    @classmethod
    def from_config(cls, config: Any) -> BPCParams:
        """Create BPCParams from a configuration object or dictionary."""
        from dataclasses import MISSING, fields

        if isinstance(config, dict):
            valid_keys = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in config.items() if k in valid_keys})

        kwargs = {f.name: getattr(config, f.name, f.default) for f in fields(cls) if f.default is not MISSING}
        # Fields with default_factory
        for f in fields(cls):
            if f.name not in kwargs:
                kwargs[f.name] = f.default_factory()  # type: ignore[misc]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert BPCParams to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
