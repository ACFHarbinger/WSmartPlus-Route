"""
Module documentation.

Attributes:
    RLAHVPLParams: Parameters for the Reinforcement Learning Augmented Hybrid Volleyball Premier League metaheuristic.

Examples:
    >>> from logic.src.policies.route_construction.learning_heuristic_algorithms import RLAHVPLParams
    >>> params = RLAHVPLParams()
    >>> print(params)
"""

# Standard library
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

# Local imports
from logic.src.configs.policies.other import RLConfig
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params import HGSParams
from logic.src.policies.route_construction.meta_heuristics.reactive_tabu_search.params import RTSParams


@dataclass
class RLAHVPLParams:
    """
    Parameters for the Reinforcement Learning Augmented Hybrid Volleyball Premier League metaheuristic.

    Combines VPL population dynamics, ACO initialization, ALNS local search, and HGS diversity
    management / crossover, all using Reinforcement Learning (RL) for adaptive operator selection.

    Attributes:
        n_teams: Number of teams.
        time_limit: Time limit for the search.
        elite_coaching_max_iterations: Maximum iterations for elite coaching.
        not_coached_max_iterations: Maximum iterations for not coached.
        coaching_acceptance_threshold: Threshold for coaching acceptance.
        sub_rate: Substitution rate.
        seed: Random seed.
        vrpp: Whether to use VRPP.
        profit_aware_operators: Whether to use profit-aware operators.
        rl_config: Reinforcement Learning configuration.
        aco_params: Ant Colony Optimization parameters.
        alns_params: Adaptive Large Neighborhood Search parameters.
        hgs_params: Hybrid Genetic Search parameters.
        tabu_no_repeat_threshold: Threshold for tabu no repeat.
        rts_params: Reactive Tabu Search parameters.
        gls_penalty_lambda: Penalty lambda for Guided Local Search.
        gls_penalty_alpha: Penalty alpha for Guided Local Search.
        gls_penalty_step: Penalty step for Guided Local Search.
        gls_probability: Probability for Guided Local Search.
    """

    # General parameters
    n_teams: int = 10
    time_limit: float = 60.0
    elite_coaching_max_iterations: int = 500
    not_coached_max_iterations: int = 100
    coaching_acceptance_threshold: float = 1e-6
    sub_rate: float = 0.2
    seed: Optional[int] = None
    vrpp: bool = False
    profit_aware_operators: bool = False

    # RL Configuration (Centralized)
    rl_config: RLConfig = field(default_factory=RLConfig)

    # ACO parameters
    aco_params: KSACOParams = field(default_factory=KSACOParams)

    # ALNS parameters
    alns_params: ALNSParams = field(default_factory=ALNSParams)

    # HGS parameters
    hgs_params: HGSParams = field(default_factory=HGSParams)

    # Tabu parameters
    tabu_no_repeat_threshold: int = 2
    rts_params: RTSParams = field(default_factory=RTSParams)

    def __post_init__(self):
        """
        Sync flags across sub-parameters.

        Args:
            None

        Returns:
            None
        """
        if self.aco_params:
            self.aco_params.vrpp = self.vrpp
            self.aco_params.profit_aware_operators = self.profit_aware_operators
        if self.alns_params:
            self.alns_params.vrpp = self.vrpp
            self.alns_params.profit_aware_operators = self.profit_aware_operators
        if self.hgs_params:
            self.hgs_params.vrpp = self.vrpp
            self.hgs_params.profit_aware_operators = self.profit_aware_operators
        if self.rts_params:
            self.rts_params.vrpp = self.vrpp
            self.rts_params.profit_aware_operators = self.profit_aware_operators

    # GLS parameters
    gls_penalty_lambda: float = 1.0
    gls_penalty_alpha: float = 0.5
    gls_penalty_step: int = 10
    gls_probability: float = 0.5

    # --- Legacy / Compatibility Properties ---
    # These properties allow existing code to access RL parameters without modification,
    # while transitioning to the new centralized RLConfig structure.

    def _get_val(self, category: str, key: str, default: Any = None) -> Any:
        """
        Get value from RL parameters.

        Args:
            category: Category of the value.
            key: Key of the value.
            default: Default value.

        Returns:
            The value.
        """
        # Resolve the category (e.g. td_learning, sarsa)
        if isinstance(self.rl_config, dict):
            cat_obj = self.rl_config.get(category, {})
        else:
            cat_obj = getattr(self.rl_config, category, None)

        # If no category found but we asked for sarsa, fallback to td_learning
        if not cat_obj and category == "sarsa":
            return self._get_val("td_learning", key, default)

        if isinstance(cat_obj, dict):
            return cat_obj.get(key, default)
        return getattr(cat_obj, key, default)

    def _get_param(self, key: str, default: Any = None) -> Any:
        """
        Get parameter from RL parameters.

        Args:
            key: Key of the parameter.
            default: Default value.

        Returns:
            The value.
        """
        if isinstance(self.rl_config, dict):
            return self.rl_config.get("params", {}).get(key, default)
        return self.rl_config.params.get(key, default)

    @property
    def bandit_algorithm(self) -> str:
        """
        Get the bandit algorithm.

        Args:
            None

        Returns:
            The bandit algorithm.
        """
        return self._get_val("bandit", "algorithm", "ucb1")

    @property
    def bandit_max_iterations(self) -> int:
        """
        Get the maximum number of iterations for the bandit solver.

        Args:
            None

        Returns:
            The maximum number of iterations for the bandit solver.
        """
        """Maximum iterations for the bandit solver."""
        return self._get_param("bandit_max_iterations", 1000)

    @property
    def bandit_quality_weight(self) -> float:
        """
        Get the weight for solution quality in bandit rewards.

        Args:
            None

        Returns:
            The weight for solution quality in bandit rewards.
        """
        return self._get_val("evolution_cmab", "quality_weight", 0.5)

    @property
    def bandit_improvement_weight(self) -> float:
        """
        Get the weight for objective improvement in bandit rewards.

        Args:
            None

        Returns:
            The weight for objective improvement in bandit rewards.
        """
        return self._get_val("evolution_cmab", "improvement_weight", 1.0)

    @property
    def bandit_diversity_weight(self) -> float:
        """
        Get the weight for population diversity in bandit rewards.

        Args:
            None

        Returns:
            The weight for population diversity in bandit rewards.
        """
        return self._get_val("evolution_cmab", "diversity_weight", 0.2)

    @property
    def bandit_novelty_weight(self) -> float:
        """
        Get the weight for solution novelty in bandit rewards.

        Args:
            None

        Returns:
            The weight for solution novelty in bandit rewards.
        """
        return self._get_val("evolution_cmab", "novelty_weight", 1.0)

    @property
    def bandit_reward_threshold(self) -> float:
        """
        Get the minimum improvement threshold for bandit rewards.

        Args:
            None

        Returns:
            The minimum improvement threshold for bandit rewards.
        """
        return self._get_val("evolution_cmab", "reward_threshold", 1e-6)

    @property
    def bandit_default_reward(self) -> float:
        """
        Get the initial reward assigned to unexplored operators.

        Args:
            None

        Returns:
            The initial reward assigned to unexplored operators.
        """
        return self._get_val("evolution_cmab", "default_reward", 5.0)

    @property
    def cfe_alpha(self) -> float:
        """
        Get the learning rate for Contextual Feature Embedding.

        Args:
            None

        Returns:
            The learning rate for Contextual Feature Embedding.
        """
        return self._get_val("context_features", "alpha", 0.1)

    @property
    def cfe_feature_dim(self) -> int:
        """
        Get the dimensionality of the context feature vector.

        Args:
            None

        Returns:
            The dimensionality of the context feature vector.
        """
        return self._get_val("context_features", "feature_dim", 8)

    @property
    def cfe_operator_selection_threshold(self) -> float:
        """
        Get the threshold for pruning low-probability operators in CFE.

        Args:
            None

        Returns:
            The threshold for pruning low-probability operators in CFE.
        """
        return self._get_val("context_features", "selection_threshold", 1e-9)

    @property
    def cfe_lambda_prior(self) -> float:
        """
        Get the prior value for the precision matrix in CFE.

        Args:
            None

        Returns:
            The prior value for the precision matrix in CFE.
        """
        return self._get_val("context_features", "lambda_prior", 1.0)

    @property
    def cfe_noise_variance(self) -> float:
        """
        Get the variance of the observation noise in CFE.

        Args:
            None

        Returns:
            The variance of the observation noise in CFE.
        """
        return self._get_val("context_features", "noise_variance", 0.1)

    @property
    def cfe_epsilon(self) -> float:
        """
        Get the exploration probability for CFE agents.

        Args:
            None

        Returns:
            The exploration probability for CFE agents.
        """
        return self._get_val("context_features", "epsilon", 0.15)

    @property
    def cfe_epsilon_decay(self) -> float:
        """
        Get the decay factor for CFE exploration.

        Args:
            None

        Returns:
            The decay factor for CFE exploration.
        """
        return self._get_val("context_features", "epsilon_decay", 0.995)

    @property
    def cfe_epsilon_decay_step(self) -> int:
        """
        Get the interval between CFE epsilon decay steps.

        Args:
            None

        Returns:
            The interval between CFE epsilon decay steps.
        """
        return self._get_val("context_features", "epsilon_decay_step", 20)

    @property
    def cfe_epsilon_min(self) -> float:
        """
        Get the minimum value for CFE exploration.

        Args:
            None

        Returns:
            The minimum value for CFE exploration.
        """
        return self._get_val("context_features", "epsilon_min", 0.05)

    @property
    def cfe_diversity_history_size(self) -> int:
        """
        Get the size of the rolling window for diversity features.

        Args:
            None

        Returns:
            The size of the rolling window for diversity features.
        """
        return self._get_val("features", "diversity_history_size", 10)

    @property
    def cfe_improvement_history_size(self) -> int:
        """
        Get the size of the rolling window for objective improvement features.

        Args:
            None

        Returns:
            The size of the rolling window for objective improvement features.
        """
        return self._get_val("features", "improvement_history_size", 10)

    @property
    def cfe_operator_reward_size(self) -> int:
        """
        Get the history size for individual operator rewards.

        Args:
            None

        Returns:
            The history size for individual operator rewards.
        """
        return self._get_param("cfe_operator_reward_size", 50)

    @property
    def cfe_improvement_threshold(self) -> float:
        """
        Get the minimum objective improvement to trigger a reward in CFE.

        Args:
            None

        Returns:
            The minimum objective improvement to trigger a reward in CFE.
        """
        return float(self._get_val("reward", "improvement_threshold", 1e-6))

    @property
    def qlearning_alpha(self) -> float:
        """
        Get the learning rate for the Q-Learning agent.

        Args:
            None

        Returns:
            The learning rate for the Q-Learning agent.
        """
        return self._get_val("td_learning", "alpha", 0.1)

    @property
    def qlearning_gamma(self) -> float:
        """
        Get the discount factor for the Q-Learning agent.

        Args:
            None

        Returns:
            The discount factor for the Q-Learning agent.
        """
        return self._get_val("td_learning", "gamma", 0.95)

    @property
    def qlearning_epsilon(self) -> float:
        """
        Get the exploration probability for the Q-Learning agent.

        Args:
            None

        Returns:
            The exploration probability for the Q-Learning agent.
        """
        return self._get_val("td_learning", "epsilon", 0.1)

    @property
    def qlearning_epsilon_decay(self) -> float:
        """
        Get the decay factor for Q-Learning exploration.

        Args:
            None

        Returns:
            The decay factor for Q-Learning exploration.
        """
        return self._get_val("td_learning", "epsilon_decay", 0.995)

    @property
    def qlearning_epsilon_decay_step(self) -> int:
        """
        Get the interval between Q-Learning epsilon decay steps.

        Args:
            None

        Returns:
            The interval between Q-Learning epsilon decay steps.
        """
        return self._get_val("td_learning", "epsilon_decay_step", 20)

    @property
    def qlearning_epsilon_min(self) -> float:
        """
        Get the minimum value for Q-Learning exploration.

        Args:
            None

        Returns:
            The minimum value for Q-Learning exploration.
        """
        return self._get_val("td_learning", "epsilon_min", 0.05)

    @property
    def qlearning_history_size(self) -> int:
        """
        Get the history buffer size for Q-Learning experience replay.

        Args:
            None

        Returns:
            The history buffer size for Q-Learning experience replay.
        """
        return self._get_val("td_learning", "history_size", 100)

    @property
    def qlearning_rewards_size(self) -> int:
        """
        Get the history buffer size for Q-Learning rewards.

        Args:
            None

        Returns:
            The history buffer size for Q-Learning rewards.
        """
        return self._get_param("qlearning_rewards_size", 100)

    @property
    def qlearning_improvement_thresholds(self) -> Tuple[float, float]:
        """
        Get the thresholds for classifying improvements in Q-Learning.

        Args:
            None

        Returns:
            The thresholds for classifying improvements in Q-Learning.
        """
        val = self._get_param("qlearning_improvement_thresholds", (1e-4, 1e-2))
        return float(val[0]), float(val[1])

    @property
    def sarsa_alpha(self) -> float:
        """
        Get the learning rate for the SARSA agent.

        Args:
            None

        Returns:
            The learning rate for the SARSA agent.
        """
        return self._get_val("sarsa", "alpha", 0.1)

    @property
    def sarsa_gamma(self) -> float:
        """
        Get the discount factor for the SARSA agent.

        Args:
            None

        Returns:
            The discount factor for the SARSA agent.
        """
        return self._get_val("sarsa", "gamma", 0.95)

    @property
    def sarsa_epsilon(self) -> float:
        """
        Get the exploration probability for the SARSA agent.

        Args:
            None

        Returns:
            The exploration probability for the SARSA agent.
        """
        return self._get_val("sarsa", "epsilon", 0.1)

    @property
    def sarsa_epsilon_decay(self) -> float:
        """
        Get the decay factor for SARSA exploration.

        Args:
            None

        Returns:
            The decay factor for SARSA exploration.
        """
        return self._get_val("sarsa", "epsilon_decay", 0.995)

    @property
    def sarsa_epsilon_decay_step(self) -> int:
        """
        Get the interval between SARSA epsilon decay steps.

        Args:
            None

        Returns:
            The interval between SARSA epsilon decay steps.
        """
        return self._get_val("sarsa", "epsilon_decay_step", 20)

    @property
    def sarsa_epsilon_min(self) -> float:
        """
        Get the minimum value for SARSA exploration.

        Args:
            None

        Returns:
            The minimum value for SARSA exploration.
        """
        return self._get_val("sarsa", "epsilon_min", 0.05)

    @property
    def sarsa_diversity_size(self) -> int:
        """
        Get the buffer size for SARSA diversity calculations.

        Args:
            None

        Returns:
            The buffer size for SARSA diversity calculations.
        """
        return self._get_param("sarsa_diversity_size", 50)

    @property
    def sarsa_scores_size(self) -> int:
        """
        Get the buffer size for SARSA score history.

        Args:
            None

        Returns:
            The buffer size for SARSA score history.
        """
        return self._get_param("sarsa_scores_size", 100)

    @property
    def sarsa_qtable_size_rate(self) -> float:
        """
        Get the learning rate multiplier for SARSA Q-Table expansion.

        Args:
            None

        Returns:
            The learning rate multiplier for SARSA Q-Table expansion.
        """
        return self._get_param("sarsa_qtable_size_rate", 0.5)

    @property
    def sarsa_improvement_thresholds(self) -> Tuple[float, float]:
        """
        Get the thresholds for classifying improvements in SARSA.

        Args:
            None

        Returns:
            The thresholds for classifying improvements in SARSA.
        """
        val = self._get_param("sarsa_improvement_thresholds", (1e-4, 1e-2))
        return float(val[0]), float(val[1])

    @property
    def sarsa_operator_progress_thresholds(self) -> Tuple[float, float]:
        """
        Get the progress thresholds for SARSA operator selection.

        Args:
            None

        Returns:
            The progress thresholds for SARSA operator selection.
        """
        val = self._get_param("sarsa_operator_progress_thresholds", (0.3, 0.7))
        return float(val[0]), float(val[1])

    @property
    def sarsa_operator_stagnation_thresholds(self) -> Tuple[int, int]:
        """
        Get the stagnation thresholds for SARSA operator selection.

        Args:
            None

        Returns:
            The stagnation thresholds for SARSA operator selection.
        """
        val = self._get_param("sarsa_operator_stagnation_thresholds", (10, 50))
        return int(val[0]), int(val[1])

    @property
    def sarsa_operator_diversity_thresholds(self) -> Tuple[float, float]:
        """
        Get the diversity thresholds for SARSA operator selection.

        Args:
            None

        Returns:
            The diversity thresholds for SARSA operator selection.
        """
        val = self._get_param("sarsa_operator_diversity_thresholds", (0.2, 0.5))
        return float(val[0]), float(val[1])
