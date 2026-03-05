# Standard library
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Local imports
from logic.src.configs.policies.other import RLConfig

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
    elite_coaching_max_iterations: int = 500
    not_coached_max_iterations: int = 100
    coaching_acceptance_threshold: float = 1e-6
    sub_rate: float = 0.2
    seed: Optional[int] = None

    # RL Configuration (Centralized)
    rl_config: RLConfig = field(default_factory=RLConfig)

    # ACO parameters
    aco_params: ACOParams = field(default_factory=ACOParams)

    # ALNS parameters
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
    gls_probability: float = 0.5

    # --- Legacy / Compatibility Properties ---
    # These properties allow existing code to access RL parameters without modification,
    # while transitioning to the new centralized RLConfig structure.

    @property
    def bandit_algorithm(self) -> str:
        """Name of the bandit algorithm to use."""
        return self.rl_config.bandit.algorithm

    @property
    def bandit_max_iterations(self) -> int:
        """Maximum iterations for the bandit."""
        return self.rl_config.params.get("bandit_max_iterations", 1000)

    @property
    def bandit_quality_weight(self) -> float:
        """Weight for solution quality in bandit reward."""
        return self.rl_config.evolution_cmab.quality_weight

    @property
    def bandit_improvement_weight(self) -> float:
        """Weight for solution improvement in bandit reward."""
        return self.rl_config.evolution_cmab.improvement_weight

    @property
    def bandit_diversity_weight(self) -> float:
        """Weight for solution diversity in bandit reward."""
        return self.rl_config.evolution_cmab.diversity_weight

    @property
    def bandit_novelty_weight(self) -> float:
        """Weight for solution novelty in bandit reward."""
        return self.rl_config.evolution_cmab.novelty_weight

    @property
    def bandit_reward_threshold(self) -> float:
        """Threshold for rewarding an operator."""
        return self.rl_config.evolution_cmab.reward_threshold

    @property
    def bandit_default_reward(self) -> float:
        """Default reward for an operator."""
        return self.rl_config.evolution_cmab.default_reward

    @property
    def cfe_alpha(self) -> float:
        """Alpha parameter for context feature extraction."""
        return self.rl_config.context_features.alpha

    @property
    def cfe_feature_dim(self) -> int:
        """Dimension of extracted features."""
        return self.rl_config.context_features.feature_dim

    @property
    def cfe_operator_selection_threshold(self) -> float:
        """Threshold for operator selection."""
        return self.rl_config.context_features.selection_threshold

    @property
    def cfe_lambda_prior(self) -> float:
        """Prior for lambda in feature extraction."""
        return self.rl_config.context_features.lambda_prior

    @property
    def cfe_noise_variance(self) -> float:
        """Noise variance for feature extraction."""
        return self.rl_config.context_features.noise_variance

    @property
    def cfe_epsilon(self) -> float:
        """Exploration rate for context-aware solvers."""
        return self.rl_config.context_features.epsilon

    @property
    def cfe_epsilon_decay(self) -> float:
        """Decay rate for epsilon."""
        return self.rl_config.context_features.epsilon_decay

    @property
    def cfe_epsilon_decay_step(self) -> int:
        """Steps between epsilon decay."""
        return self.rl_config.context_features.epsilon_decay_step

    @property
    def cfe_epsilon_min(self) -> float:
        """Minimum epsilon value."""
        return self.rl_config.context_features.epsilon_min

    @property
    def cfe_diversity_history_size(self) -> int:
        """History size for diversity tracking."""
        return self.rl_config.features.diversity_history_size

    @property
    def cfe_improvement_history_size(self) -> int:
        """History size for improvement tracking."""
        return self.rl_config.features.improvement_history_size

    @property
    def cfe_operator_reward_size(self) -> int:
        """History size for operator rewards."""
        return self.rl_config.params.get("cfe_operator_reward_size", 50)

    @property
    def cfe_improvement_threshold(self) -> float:
        """Threshold for considering an improvement significant."""
        return self.rl_config.reward.improvement_threshold

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
    def qlearning_history_size(self) -> int:
        """History size for Q-Learning."""
        return self.rl_config.td_learning.history_size

    @property
    def qlearning_rewards_size(self) -> int:
        """Size of rewards history."""
        return self.rl_config.params.get("qlearning_rewards_size", 100)

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
    def sarsa_scores_size(self) -> int:
        """History size for scores in SARSA."""
        return self.rl_config.params.get("sarsa_scores_size", 100)

    @property
    def sarsa_qtable_size_rate(self) -> float:
        """Size rate for the Q-Table."""
        return self.rl_config.params.get("sarsa_qtable_size_rate", 0.5)

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
