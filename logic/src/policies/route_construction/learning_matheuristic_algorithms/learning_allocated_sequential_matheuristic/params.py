r"""Configuration parameters for the LBBD → ALNS → BPC → RL → SP pipeline.

Architecture overview
---------------------
Stage 1  LBBD Master   — Knapsack-style MIP selects which customer nodes to visit.
                         Receives Benders cuts from Stage 2 and RL-adjusted cut
                         budgets from Stage 4.
Stage 2  LBBD Sub      — Routing subproblem solved per master selection.
                         Cut generation: no-good / optimality / Pareto-optimal
                         (Magnanti-Wang 1981) Benders cuts fed back to master.
Stage 3  ALNS          — Repairs and extends the LBBD incumbent using the
                         project's existing ALNSSolver with pool harvesting.
Stage 4  BPC           — Exact column generation with SP incumbent seeding.
Stage 5  RL controller — Online bandit (LinUCB) or offline PPO policy that
                         allocates time budgets across stages and selects
                         cut families, ng-sizes, and ALNS operator weights.
Stage 6  SP merge      — Set-Partitioning MIP over the global route pool.

Quality / speed dial
--------------------
A single ``alpha ∈ [0, 1]`` controls the default time allocation:

    alpha = 0.0  →  LBBD + tiny ALNS only  (fastest, no BPC)
    alpha = 0.5  →  balanced               (default)
    alpha = 1.0  →  full BPC + ALNS + deep LBBD  (highest quality, slowest)

The RL controller (when enabled) *overrides* the alpha-derived defaults
after collecting enough experience across instances.

Attributes:
    LASMPipelineParams: Dataclass holding all solver configuration.

Example:
    >>> p = LASMPipelineParams(alpha=0.5, time_limit=120.0)
    >>> budgets = p.stage_budgets()
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LASMPipelineParams:
    """Configuration for the LBBD → ALNS → BPC → RL → SP pipeline.

    Attributes
    ----------
    alpha : float
        Quality/speed dial ∈ [0, 1].
    time_limit : float
        Total wall-clock budget in seconds.
    seed : int or None
        Global RNG seed.

    LBBD master / sub
    -----------------
    lbbd_max_iterations : int
        Maximum outer Benders iterations (master–sub cycles).
    lbbd_master_time_frac : float
        Fraction of stage budget given to each master solve.
    lbbd_sub_solver : str
        Routing sub-solver: 'alns', 'bpc', or 'greedy'.
    lbbd_cut_families : list[str]
        Active Benders cut types: any combination of
        'nogood', 'optimality', 'pareto', 'combinatorial'.
    lbbd_pareto_eps : float
        Tolerance ε for Magnanti-Wang Pareto-optimal cut selection.
    lbbd_sub_time_frac : float
        Fraction of stage budget given to each sub-problem solve.
    lbbd_min_cover_ratio : float
        Minimum fraction of mandatory nodes the master must cover.
    lbbd_use_warm_cuts : bool
        If True, carry Benders cuts forward across RL episodes
        (cross-instance cut pool).

    ALNS
    ----
    alns_max_iterations : int
        Hard iteration cap (0 = derive from alpha).
    alns_segment_size : int
        Ropke–Pisinger weight-update segment size.
    alns_reaction_factor : float
        Operator weight learning rate r.
    alns_cooling_rate : float
        SA temperature decay per iteration.
    alns_start_temp_control : float
        Acceptance probability at start temperature for w%-worse solutions.
    alns_sigma_1 / sigma_2 / sigma_3 : float
        Scoring constants for new-best / better / accepted-worse solutions.
    alns_xi : float
        Fraction of n for the removal upper-bound cap.
    alns_min_removal : int
        Minimum nodes removed per destroy step.
    alns_noise_factor : float
        Noise amplitude η for noisy repair operators.
    alns_worst_removal_randomness : float
        Randomness exponent p ≥ 1 for worst removal.
    alns_shaw_randomization : float
        Shaw relatedness randomisation factor.
    alns_regret_pool : str
        Regret variants: 'regret2', 'regret234', or 'regretAll'.
    alns_extended_operators : bool
        Add string/cluster/neighbor destroy operators.
    alns_profit_aware_operators : bool
        Use profit-aware operator variants.
    alns_vrpp : bool
        Allow insertion from full candidate pool.
    alns_engine : str
        ALNS backend ('custom', 'package', 'ortools').

    BPC
    ---
    bpc_ng_size_min / bpc_ng_size_max : int
        ng-neighborhood size range, interpolated by alpha.
    bpc_max_bb_nodes_min / bpc_max_bb_nodes_max : int
        B&B node cap range, interpolated by alpha.
    bpc_cutting_planes : str
        Cut family ('rcc', 'saturated_arc_lci', 'all').
    bpc_branching_strategy : str
        Branching rule ('divergence', 'ryan_foster', 'edge').
    skip_bpc : bool
        Force-skip the BPC stage.

    RL controller
    -------------
    rl_mode : str
        'online'   — LinUCB bandit, learns within a single run.
        'offline'  — Pretrained PPO policy loaded from rl_policy_path.
        'hybrid'   — Offline policy + online fine-tuning.
        'disabled' — No RL; use alpha-derived budgets only.
    rl_policy_path : str or None
        Path to a serialised PPO/offline policy (pickle or JSON).
    rl_exploration : float
        LinUCB exploration coefficient λ (δ in Li et al. 2010).
    rl_window : int
        Sliding-window size for non-stationary reward tracking.
        Set 0 to use all history (stationary assumption).
    rl_min_samples : int
        Minimum decisions before RL overrides alpha-derived budgets.
    rl_reward_shaping : str
        How profit improvement is shaped into a reward signal.
        'absolute'  — reward = Δprofit.
        'relative'  — reward = Δprofit / (|best_profit| + 1).
        'efficiency'— reward = Δprofit / Δtime.
    rl_state_features : list[str]
        Which instance features form the RL context vector.
        Options: 'n_nodes', 'fill_mean', 'fill_std', 'mandatory_ratio',
                 'lp_gap', 'pool_size', 'time_remaining', 'alpha',
                 'iter_count'.
    rl_action_space : str
        'budgets'   — action is a time-budget split across stages.
        'operators' — action is ALNS operator weight multipliers.
        'combined'  — both budgets and operator weights.
    rl_discount : float
        Discount factor γ for multi-step return (offline PPO only).

    SP merge
    --------
    sp_pool_cap : int
        Maximum routes in the SP MIP.
    sp_mip_gap : float
        Relative MIP gap for early termination.
    """

    # ── Quality / speed dial ───────────────────────────────────────────────
    alpha: float = 0.5
    time_limit: float = 120.0
    seed: Optional[int] = None

    # ── LBBD ──────────────────────────────────────────────────────────────
    lbbd_max_iterations: int = 20
    lbbd_master_time_frac: float = 0.15
    lbbd_sub_solver: str = "alns"  # 'alns' | 'bpc' | 'greedy'
    lbbd_cut_families: List[str] = None  # type: ignore[assignment]
    lbbd_pareto_eps: float = 1e-4
    lbbd_sub_time_frac: float = 0.70
    lbbd_min_cover_ratio: float = 1.0
    lbbd_use_warm_cuts: bool = False

    # ── ALNS ──────────────────────────────────────────────────────────────
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

    # ── BPC ───────────────────────────────────────────────────────────────
    bpc_ng_size_min: int = 8
    bpc_ng_size_max: int = 16
    bpc_max_bb_nodes_min: int = 200
    bpc_max_bb_nodes_max: int = 1000
    bpc_cutting_planes: str = "rcc"
    bpc_branching_strategy: str = "divergence"
    skip_bpc: bool = False

    # ── RL controller ─────────────────────────────────────────────────────
    rl_mode: str = "online"  # 'online'|'offline'|'hybrid'|'disabled'
    rl_policy_path: Optional[str] = None
    rl_exploration: float = 1.0
    rl_window: int = 20
    rl_min_samples: int = 5
    rl_reward_shaping: str = "efficiency"
    rl_state_features: List[str] = None  # type: ignore[assignment]
    rl_action_space: str = "budgets"
    rl_discount: float = 0.99

    # ── SP merge ──────────────────────────────────────────────────────────
    sp_pool_cap: int = 50_000
    sp_mip_gap: float = 1e-4

    def __post_init__(self) -> None:
        if self.lbbd_cut_families is None:
            self.lbbd_cut_families = ["nogood", "optimality", "pareto"]
        if self.rl_state_features is None:
            self.rl_state_features = [
                "n_nodes",
                "fill_mean",
                "fill_std",
                "mandatory_ratio",
                "lp_gap",
                "pool_size",
                "time_remaining",
                "alpha",
            ]

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def stage_budgets(
        self,
        budget_override: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float, float, float]:
        """Compute per-stage time budgets.

        When the RL controller provides a ``budget_override`` dict it replaces
        the alpha-derived fractions.  The five values always sum to time_limit.

        Args:
            budget_override: Optional dict with keys 'lbbd', 'alns', 'bpc',
                'sp' giving explicit fractions ∈ (0, 1).  Missing keys fall
                back to alpha-derived defaults.

        Returns:
            (tau_lbbd, tau_alns, tau_bpc, tau_rl_overhead, tau_sp)
        """
        T = self.time_limit
        a = max(0.0, min(1.0, self.alpha))

        # RL overhead is always small (bookkeeping, not solver time)
        tau_rl_overhead = min(2.0, T * 0.01)

        if budget_override:
            tau_lbbd = T * budget_override.get("lbbd", max(0.05, 0.18 - 0.08 * a))
            tau_alns = T * budget_override.get("alns", 0.20 + 0.15 * a)
            tau_bpc = T * budget_override.get("bpc", 0.00 + 0.60 * a)
            tau_sp = T * budget_override.get("sp", 0.05)
        else:
            tau_lbbd = T * max(0.05, 0.18 - 0.08 * a)
            tau_alns = T * (0.20 + 0.15 * a)
            tau_bpc = T * (0.00 + 0.60 * a)
            tau_sp = min(30.0, T * 0.05)

        total = tau_lbbd + tau_alns + tau_bpc + tau_rl_overhead + tau_sp
        s = T / total if total > 0 else 1.0
        return (
            tau_lbbd * s,
            tau_alns * s,
            tau_bpc * s,
            tau_rl_overhead * s,
            tau_sp * s,
        )

    def alns_iterations(self) -> int:
        """Effective ALNS iteration count."""
        if self.alns_max_iterations > 0:
            return self.alns_max_iterations
        return max(500, int(2_000 + 18_000 * max(0.0, min(1.0, self.alpha))))

    def bpc_ng_size(self) -> int:
        a = max(0.0, min(1.0, self.alpha))
        return self.bpc_ng_size_min + int(a * (self.bpc_ng_size_max - self.bpc_ng_size_min))

    def bpc_max_bb_nodes(self) -> int:
        a = max(0.0, min(1.0, self.alpha))
        return self.bpc_max_bb_nodes_min + int(a * (self.bpc_max_bb_nodes_max - self.bpc_max_bb_nodes_min))

    def as_alns_values_dict(self) -> Dict[str, Any]:
        """Build the flat dict expected by the existing ALNSParams.from_config."""
        return {
            "engine": self.alns_engine,
            "time_limit": 0.0,
            "max_iterations": self.alns_iterations(),
            "start_temp": 0.0,
            "cooling_rate": self.alns_cooling_rate,
            "reaction_factor": self.alns_reaction_factor,
            "min_removal": self.alns_min_removal,
            "start_temp_control": self.alns_start_temp_control,
            "xi": self.alns_xi,
            "segment_size": self.alns_segment_size,
            "noise_factor": self.alns_noise_factor,
            "worst_removal_randomness": self.alns_worst_removal_randomness,
            "shaw_randomization": self.alns_shaw_randomization,
            "max_removal_cap": 100,
            "regret_pool": self.alns_regret_pool,
            "sigma_1": self.alns_sigma_1,
            "sigma_2": self.alns_sigma_2,
            "sigma_3": self.alns_sigma_3,
            "vrpp": self.alns_vrpp,
            "profit_aware_operators": self.alns_profit_aware_operators,
            "extended_operators": self.alns_extended_operators,
            "seed": self.seed,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "LASMPipelineParams":
        """Construct from a dict or attribute-bearing object.

        Unknown keys are silently ignored.
        """
        if config is None:
            return cls()
        valid = {f.name for f in fields(cls)}
        raw: Dict[str, Any] = {}
        if isinstance(config, dict):
            raw = {k: v for k, v in config.items() if k in valid}
        else:
            for f in fields(cls):
                if hasattr(config, f.name):
                    raw[f.name] = getattr(config, f.name)
        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
