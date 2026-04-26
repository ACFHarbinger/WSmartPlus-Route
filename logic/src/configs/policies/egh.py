"""
Exact Guided Heuristic (EGH) configuration.

Attributes:
    ExactGuidedHeuristicConfig: Attributes for EGH configuration.

Example:
    >>> from logic.src.configs.policies.egh import ExactGuidedHeuristicConfig
    >>> config = ExactGuidedHeuristicConfig()
    >>> config.alpha
    0.5
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExactGuidedHeuristicConfig:
    """Configuration for Exact Guided Heuristic (EGH) policy.

    Attributes:
        alpha: Quality/speed dial ∈ [0, 1].
            0.0 = TCF + tiny ALNS only (no BPC stage).
            0.5 = balanced (default).
            1.0 = full BPC + large ALNS (highest quality, slowest).
        time_limit: Total wall-clock budget in seconds.
        seed: Global RNG seed for reproducibility.

        # ALNS overrides
        alns_max_iterations: Hard iteration cap for ALNS.
            0 = derive from alpha: max(500, 2000 + 18000 * alpha).
        alns_segment_size: Weight-update segment size (Ropke & Pisinger 2006).
        alns_reaction_factor: Learning rate r for weight updates.
        alns_cooling_rate: SA temperature decay factor per iteration.
        alns_start_temp_control: 'w' parameter — accept a solution 'w*100 %'
            worse than current with probability 0.5 at start temperature.
        alns_sigma_1: Score awarded for a new global-best solution.
        alns_sigma_2: Score awarded for a better-than-current new solution.
        alns_sigma_3: Score awarded for an accepted worse new solution.
        alns_xi: Fraction of n for the removal upper-bound cap.
        alns_min_removal: Minimum nodes removed per destroy step.
        alns_noise_factor: Noise amplitude η for noisy repair operators.
        alns_worst_removal_randomness: Randomness exponent p ≥ 1 for worst removal.
        alns_shaw_randomization: Shaw randomisation factor p_shaw.
        alns_regret_pool: Which regret variants to use
            ('regret2', 'regret234', 'regretAll').
        alns_extended_operators: If True, add string/cluster/neighbor destroy
            operators.
        alns_profit_aware_operators: If True, use profit-aware operator variants.
        alns_vrpp: If True, allow ALNS repair operators to insert nodes from the
            full candidate pool.
        alns_engine: Which ALNS backend to use ('custom', 'package', 'ortools').

        # BPC overrides
        bpc_ng_size_min: Minimum ng-neighborhood size (used when alpha=0).
        bpc_ng_size_max: Maximum ng-neighborhood size (used when alpha=1).
        bpc_max_bb_nodes_min: Minimum B&B node cap (alpha=0).
        bpc_max_bb_nodes_max: Maximum B&B node cap (alpha=1).
        bpc_cutting_planes: Cut family for BPC ('rcc', 'saturated_arc_lci',
            'all', etc.).
        bpc_branching_strategy: Branching rule ('divergence', 'ryan_foster',
            'edge').
        skip_bpc: Force-skip the BPC stage regardless of alpha or time budget.

        # SP-merge overrides
        sp_pool_cap: Maximum number of routes kept in the SP-merge MIP.
        sp_mip_gap: Relative gap at which the SP MIP is considered solved.
    """

    # Quality / speed dial
    alpha: float = 0.5
    time_limit: float = 120.0
    seed: Optional[int] = None

    # ALNS
    alns_max_iterations: int = 0
    alns_segment_size: int = 100
    alns_reaction_factor: float = 0.1
    alns_cooling_rate: float = 0.995
    alns_start_temp_control: float = 0.05
    alns_sigma_1: float = 33.0
    alns_sigma_2: float = 9.0
    alns_sigma_3: float = 13.0
    alns_xi: float = 0.4
    alns_min_removal: int = 4
    alns_noise_factor: float = 0.025
    alns_worst_removal_randomness: float = 3.0
    alns_shaw_randomization: float = 6.0
    alns_regret_pool: str = "regret234"
    alns_extended_operators: bool = False
    alns_profit_aware_operators: bool = True
    alns_vrpp: bool = True
    alns_engine: str = "custom"

    # BPC
    bpc_ng_size_min: int = 8
    bpc_ng_size_max: int = 16
    bpc_max_bb_nodes_min: int = 200
    bpc_max_bb_nodes_max: int = 1000
    bpc_cutting_planes: str = "rcc"
    bpc_branching_strategy: str = "divergence"
    skip_bpc: bool = False

    # SP merge
    sp_pool_cap: int = 50_000
    sp_mip_gap: float = 1e-4
