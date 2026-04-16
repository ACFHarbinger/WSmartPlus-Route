"""
Configuration dataclass for the Multi-Period Integer L-Shaped (MP-ILS-BD) policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MPILSBDConfig:
    """Configuration for the Two-Stage Stochastic MPVRP via Integer L-Shaped method.

    Extends the standard ILS-BD framework with explicit inventory balance
    linking constraints across T periods:

        I_{it} = I_{i,t-1} + d_{it}(ξ̄) - q_{it}   ∀i ∈ V, t ∈ T
        q_{it} ≤ M · y_{it}                          ∀i ∈ V, t ∈ T

    Attributes:
        time_limit: Total solver time limit (seconds).
        master_time_limit: Time limit for each master problem solve (seconds).
        mip_gap: MIP optimality gap tolerance.
        max_iterations: Maximum Benders outer iterations.
        theta_lower_bound: Lower bound on the recourse surrogate θ.
        max_cuts_per_round: Maximum constraint cuts per callback round.
        enable_heuristic_rcc_separation: Use heuristic capacity cuts.
        enable_comb_cuts: Enable comb inequality separation.
        verbose: Enable Gurobi output.
        horizon: Planning horizon T (number of days).
        stockout_penalty: Penalty per unit of bin overflow per day.
        big_m: Big-M constant for coupling constraint q_{it} ≤ M · y_{it}.
        mean_scenario_only: If True, use mean demand for balance constraints;
            stochastic recourse handles uncertainty in the subproblem.
        initial_inventory: Initial fill level (%) for all bins (default 0).
        seed: Random seed (for scenario sampling in the subproblem).
    """

    # --- Standard ILS-BD parameters ---
    time_limit: float = 300.0
    master_time_limit: float = 60.0
    mip_gap: float = 0.01
    max_iterations: int = 50
    theta_lower_bound: float = -1e6
    max_cuts_per_round: int = 10
    enable_heuristic_rcc_separation: bool = True
    enable_comb_cuts: bool = False
    verbose: bool = False

    # --- Multi-period extensions ---
    horizon: int = 7
    stockout_penalty: float = 500.0
    big_m: float = 1e4
    mean_scenario_only: bool = True
    initial_inventory: float = 0.0
    seed: Optional[int] = None
