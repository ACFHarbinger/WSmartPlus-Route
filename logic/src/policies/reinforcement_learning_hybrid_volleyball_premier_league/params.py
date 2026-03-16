"""
Hyperparameters for Reinforcement Learning Hybrid Volleyball Premier League (RL-HVPL).

This configuration bridges the basic HVPL (ACO + ALNS in population framework)
and the advanced RL-AHVPL (with HGS, CMAB, GLS), providing RL-enhanced operators
without the full genetic evolution complexity.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from logic.src.configs.policies.other import RLConfig

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..ant_colony_optimization_k_sparse.params import ACOParams


@dataclass
class RLHVPLParams:
    """
    Parameters for the Reinforcement Learning Hybrid Volleyball Premier League.

    Combines:
    - ACO with Q-Learning for intelligent construction
    - ALNS with SARSA for adaptive destroy/repair operator selection
    - Population-based framework (VPL) for global search
    - NO genetic operators (simpler than RL-AHVPL)
    """

    # ===== General Parameters =====
    n_teams: int = 10  # Population size
    max_iterations: int = 100  # Number of league seasons
    sub_rate: float = 0.2  # Fraction of teams replaced per iteration
    time_limit: float = 60.0  # Overall time limit in seconds
    seed: Optional[int] = None

    # ===== RL Configuration (Centralized) =====
    rl_config: RLConfig = field(default_factory=RLConfig)

    # ===== ACO Parameters (with Q-Learning) =====
    aco_params: ACOParams = field(
        default_factory=lambda: ACOParams(
            n_ants=10,
            k_sparse=10,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            q0=0.9,
            tau_0=None,
            tau_min=0.001,
            tau_max=10.0,
            max_iterations=1,  # Single construction per call
            time_limit=10.0,
            local_search=True,
            local_search_iterations=50,
            elitist_weight=1.0,
        )
    )

    # ===== ALNS Parameters (with SARSA) =====
    alns_params: ALNSParams = field(
        default_factory=lambda: ALNSParams(
            max_iterations=200,
            start_temp=100.0,
            cooling_rate=0.97,
            reaction_factor=0.5,
            min_removal=1,
            max_removal_pct=0.3,
            time_limit=20.0,
        )
    )

    # ===== Pheromone Update Strategy =====
    pheromone_update_strategy: str = "profit"  # "profit" or "cost"
    profit_weight: float = 1.0  # Weight for profit-based pheromone updates

    # ===== Coaching Parameters =====
    elite_coaching_iterations: int = 300  # Intensive coaching for top teams
    regular_coaching_iterations: int = 100  # Light coaching for others
    elite_size: int = 3  # Number of teams receiving elite coaching

    # --- Legacy / Compatibility Properties ---
    # These properties allow existing code to access RL parameters without modification,
    # maintaining consistency with RL-AHVPL interface.

    @property
    def qlearning_alpha(self) -> float:
        """Learning rate for Q-Learning."""
        return self.rl_config.td_learning.alpha

    @property
    def qlearning_gamma(self) -> float:
        """Discount factor for Q-Learning."""
        return self.rl_config.td_learning.gamma

    @property
    def qlearning_epsilon(self) -> float:
        """Exploration rate for Q-Learning."""
        return self.rl_config.td_learning.epsilon

    @property
    def qlearning_epsilon_decay(self) -> float:
        """Decay rate for epsilon."""
        return self.rl_config.td_learning.epsilon_decay

    @property
    def qlearning_epsilon_decay_step(self) -> int:
        """Steps between epsilon decay."""
        return self.rl_config.td_learning.epsilon_decay_step

    @property
    def qlearning_epsilon_min(self) -> float:
        """Minimum epsilon value."""
        return self.rl_config.td_learning.epsilon_min

    @property
    def qlearning_improvement_thresholds(self) -> Tuple[float, float]:
        """Thresholds for improvement classification."""
        return self.rl_config.params.get("qlearning_improvement_thresholds", (1e-4, 1e-2))

    @property
    def sarsa_alpha(self) -> float:
        """Learning rate for SARSA."""
        return (self.rl_config.sarsa or self.rl_config.td_learning).alpha

    @property
    def sarsa_gamma(self) -> float:
        """Discount factor for SARSA."""
        return (self.rl_config.sarsa or self.rl_config.td_learning).gamma

    @property
    def sarsa_epsilon(self) -> float:
        """Exploration rate for SARSA."""
        return (self.rl_config.sarsa or self.rl_config.td_learning).epsilon

    @property
    def sarsa_epsilon_decay(self) -> float:
        """Decay rate for epsilon."""
        return (self.rl_config.sarsa or self.rl_config.td_learning).epsilon_decay

    @property
    def sarsa_epsilon_decay_step(self) -> int:
        """Steps between epsilon decay."""
        return (self.rl_config.sarsa or self.rl_config.td_learning).epsilon_decay_step

    @property
    def sarsa_epsilon_min(self) -> float:
        """Minimum epsilon value."""
        return (self.rl_config.sarsa or self.rl_config.td_learning).epsilon_min

    @property
    def sarsa_diversity_size(self) -> int:
        """History size for diversity tracking in SARSA."""
        return self.rl_config.params.get("sarsa_diversity_size", 50)

    @property
    def sarsa_improvement_thresholds(self) -> Tuple[float, float]:
        """Thresholds for improvement classification in SARSA."""
        return self.rl_config.params.get("sarsa_improvement_thresholds", (1e-4, 1e-2))

    @property
    def sarsa_operator_progress_thresholds(self) -> Tuple[float, float]:
        """Thresholds for progress classification."""
        return self.rl_config.params.get("sarsa_operator_progress_thresholds", (0.3, 0.7))

    @property
    def sarsa_operator_stagnation_thresholds(self) -> Tuple[int, int]:
        """Thresholds for stagnation classification."""
        return self.rl_config.params.get("sarsa_operator_stagnation_thresholds", (10, 50))

    @property
    def sarsa_operator_diversity_thresholds(self) -> Tuple[float, float]:
        """Thresholds for diversity classification."""
        return self.rl_config.params.get("sarsa_operator_diversity_thresholds", (0.2, 0.5))
