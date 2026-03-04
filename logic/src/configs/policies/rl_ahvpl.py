"""
RL-AHVPL (Reinforcement Learning Augmented Hybrid Volleyball Premier League) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .aco import ACOConfig
from .alns import ALNSConfig
from .hgs import HGSConfig
from .rts import RTSConfig


@dataclass
class RLAHVPLConfig:
    """
    Configuration for the Reinforcement Learning Augmented Hybrid Volleyball Premier League policy.

    Combines VPL population dynamics, ACO initialization, ALNS local search, and HGS diversity
    management / crossover, all using Reinforcement Learning (RL) for adaptive operator selection.
    """

    engine: str = "rl_ahvpl"

    # General parameters
    n_teams: int = 10
    time_limit: float = 60.0
    max_iterations: int = 1000
    elite_coaching_max_iterations: int = 500
    not_coached_max_iterations: int = 100
    coaching_acceptance_threshold: float = 1e-6
    sub_rate: float = 0.2

    # Nested component configs
    aco: ACOConfig = field(default_factory=ACOConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)
    hgs: HGSConfig = field(default_factory=HGSConfig)
    rts: RTSConfig = field(default_factory=RTSConfig)

    # Tabu parameters
    tabu_no_repeat_threshold: int = 2

    # GLS parameters
    gls_penalty_lambda: float = 1.0
    gls_penalty_alpha: float = 0.5
    gls_penalty_step: int = 10
    gls_probability: float = 0.5

    # Contextual Multi-Armed Bandit parameters
    bandit_algorithm: str = "linucb"
    bandit_max_iterations: int = 1000
    bandit_quality_weight: float = 0.5
    bandit_improvement_weight: float = 1.0
    bandit_diversity_weight: float = 0.2
    bandit_novelty_weight: float = 1.0
    bandit_reward_threshold: float = 1e-6
    bandit_default_reward: float = 5.0

    # Context Feature Extractor parameters
    cfe_alpha: float = 0.1
    cfe_feature_dim: int = 8
    cfe_operator_selection_threshold: float = 1e-9
    cfe_lambda_prior: float = 1.0
    cfe_noise_variance: float = 0.1
    cfe_epsilon: float = 0.15
    cfe_epsilon_decay: float = 0.995
    cfe_epsilon_decay_step: int = 20
    cfe_epsilon_min: float = 0.05
    cfe_diversity_history_size: int = 10
    cfe_improvement_history_size: int = 10
    cfe_operator_reward_size: int = 50
    cfe_improvement_threshold: float = 1e-6

    # Q-Learning parameters
    qlearning_alpha: float = 0.1
    qlearning_gamma: float = 0.9
    qlearning_epsilon: float = 0.1
    qlearning_epsilon_decay: float = 0.99
    qlearning_epsilon_decay_step: int = 10
    qlearning_epsilon_min: float = 0.05
    qlearning_history_size: int = 10
    qlearning_rewards_size: int = 20
    qlearning_improvement_thresholds: List[float] = field(default_factory=lambda: [1e-4, -1e-4])

    # SARSA parameters
    sarsa_alpha: float = 0.1
    sarsa_gamma: float = 0.95
    sarsa_epsilon: float = 0.15
    sarsa_epsilon_decay: float = 0.995
    sarsa_epsilon_decay_step: int = 50
    sarsa_epsilon_min: float = 0.05
    sarsa_diversity_size: int = 10
    sarsa_scores_size: int = 50
    sarsa_qtable_size_rate: float = 0.5
    sarsa_improvement_thresholds: List[float] = field(default_factory=lambda: [-1e-6, 1e-6])
    sarsa_operator_progress_thresholds: List[float] = field(default_factory=lambda: [0.33, 0.67])
    sarsa_operator_stagnation_thresholds: List[int] = field(default_factory=lambda: [10, 30])
    sarsa_operator_diversity_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.7])

    # Common policy fields
    vrpp: bool = True
    seed: Optional[int] = None
    must_go: List[str] = field(default_factory=list)
    post_processing: List[Any] = field(default_factory=list)
