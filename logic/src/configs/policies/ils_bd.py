"""
Benders (Integer L-Shaped) policy configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class IntegerLShapedBendersConfig:
    """Configuration for the Integer L-Shaped (Benders Decomposition) policy.

    The solver formulates the routing problem as a Two-Stage Stochastic Integer
    Program (2-SIP) and solves it via iterative Benders decomposition.

    Attributes:
        time_limit: Maximum wall-clock seconds for the overall Benders solve.
        n_scenarios: Number of SAA discrete scenarios for recourse approximation.
        seed: Random seed for reproducible scenario generation.
        vrpp: Whether to formulate as VRPP (customer nodes may be skipped).
        profit_aware_operators: Whether to use profit-aware warm-start heuristics.
        max_benders_iterations: Maximum outer Benders (L-shaped) iterations.
        benders_gap: Convergence tolerance for the outer Benders loop.
        overflow_penalty: Penalty per %-fill unit that overflows an unvisited bin
            above the collection threshold (€/%-fill).
        undervisit_penalty: Penalty per %-fill unit below the collection threshold
            for visited bins — cost of a wasted trip (€/%-fill).
        collection_threshold: Fill level % at which overflow/undervisit penalties
            are triggered (τ in the mathematical formulation).
        fill_rate_cv: Coefficient of variation for SAA Gamma scenario generation.
        mip_gap: Relative MIP gap for each Gurobi master problem solve.
        theta_lower_bound: Initial lower bound on the surrogate variable θ.
        verbose: Enable solver output and per-iteration Benders logging.
        max_cuts_per_round: Maximum SECs / RCCs added per Gurobi callback.
        enable_heuristic_rcc_separation: Enable heuristic RCC separation.
        enable_comb_cuts: Enable heuristic comb inequality separation.
        must_go: List of must-go strategy configuration files.
        post_processing: List of post-processing operations to apply.
    """

    time_limit: float = 120.0
    n_scenarios: int = 20
    seed: Optional[int] = 42
    vrpp: bool = True
    profit_aware_operators: bool = False
    max_benders_iterations: int = 50
    benders_gap: float = 1e-4
    overflow_penalty: float = 100.0
    undervisit_penalty: float = 10.0
    collection_threshold: float = 70.0
    fill_rate_cv: float = 0.3
    mip_gap: float = 0.01
    theta_lower_bound: float = 0.0
    verbose: bool = False
    max_cuts_per_round: int = 50
    enable_heuristic_rcc_separation: bool = True
    enable_comb_cuts: bool = False
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
