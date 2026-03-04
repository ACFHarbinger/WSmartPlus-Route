"""
Hyperparameters for the Reinforcement Learning Augmented Hybrid Volleyball Premier League (RL-AHVPL) algorithm.

Extends the base AHVPL framework with Reinforcement Learning (RL) integration
for adaptive selection of crossover, local search, and ruin and repair operators.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..ant_colony_optimization.k_sparse_aco.params import ACOParams
from ..hybrid_genetic_search.params import HGSParams
from ..reactive_tabu_search.params import RTSParams


@dataclass
class RLAHVPLParams:
    """
    Parameters for the Reinforcement Learning Augmented Hybrid Volleyball Premier League metaheuristic.

    Combines VPL population dynamics, ACO initialization, ALNS local search, and HGS diversity
    management / crossover, all using Reinforcement Learning (RL) for adaptive operator selection.
    """

    # General parameters
    n_teams: int = 10
    time_limit: float = 60.0
    max_iterations: int = 1000
    vns_max_iterations: int = 500
    sub_rate: float = 0.2
    max_no_improvement: int = 20
    seed: Optional[int] = None

    # ACO parameters
    aco_params: ACOParams = field(default_factory=ACOParams)

    # ALNS parameters
    not_coached_alns_iterations: int = 100
    alns_params: ALNSParams = field(default_factory=ALNSParams)

    # HGS parameters
    hgs_params: HGSParams = field(default_factory=HGSParams)

    # Tabu parameters
    tabu_no_repeat_threshold: int = 2
    rts_params: RTSParams = field(default_factory=RTSParams)

    # GLS parameters
    gls_penalty_lambda: float = 1.0
    gls_penalty_alpha: float = 0.5
    gls_penalty_step: int = 10

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
    qlearning_improvement_thresholds: List[float] = [1e-4, -1e-4]

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
    sarsa_improvement_thresholds: List[float] = [-1e-6, 1e-6]
    sarsa_operator_progress_thresholds: List[float] = [0.33, 0.67]
    sarsa_operator_stagnation_thresholds: List[int] = [10, 30]
    sarsa_operator_diversity_thresholds: List[float] = [0.3, 0.7]
