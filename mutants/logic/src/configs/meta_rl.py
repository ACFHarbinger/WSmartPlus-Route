"""
Meta-RL Config module.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MetaRLConfig:
    """Meta-RL and HRL algorithm configuration.

    Attributes:
        use_meta: Whether to use meta-learning wrapper.
        meta_strategy: Meta-learning strategy ('rnn', 'bandit', 'morl', etc.).
        meta_lr: Learning rate for meta-optimizer.
        meta_hidden_dim: Hidden dimension for meta-network.
        meta_history_length: History length for meta-learning.
        hrl_threshold: HRL threshold parameter.
    """

    # Meta-RL
    use_meta: bool = False
    meta_strategy: str = "rnn"  # rnn|bandit|morl|tdl|hypernet|hrl
    meta_lr: float = 1e-3
    meta_hidden_dim: int = 64
    meta_history_length: int = 10
    mrl_exploration_factor: float = 2.0
    mrl_range: List[float] = field(default_factory=lambda: [0.01, 5.0])
    mrl_batch_size: int = 256
    mrl_step: int = 10

    # HRL
    hrl_threshold: float = 0.9
    hrl_epochs: int = 4
    hrl_clip_eps: float = 0.2
    hrl_pid_target: float = 0.0003
    hrl_lambda_waste: float = 300.0
    hrl_lambda_cost: float = 0.5
    hrl_lambda_overflow_initial: float = 2000.0
    hrl_lambda_overflow_min: float = 100.0
    hrl_lambda_overflow_max: float = 5000.0
    hrl_lambda_pruning: float = 0.5
    hrl_lambda_mask_aux: float = 5.0
    hrl_entropy_coef: float = 0.01
    shared_encoder: bool = True
    gat_hidden_dim: int = 128
    lstm_hidden_dim: int = 64
    gate_prob_threshold: float = 0.5
    lr_critic_value: float = 1e-4

    # Contextual Bandits
    cb_exploration_method: str = "ucb"
    cb_num_configs: int = 10
    cb_epsilon_decay: float = 0.995
    cb_min_epsilon: float = 0.01
    cb_context_features: List[str] = field(
        default_factory=lambda: ["waste", "overflow", "length", "visited_ratio", "day"]
    )
    cb_features_aggregation: str = "avg"

    # MORL
    morl_objectives: List[str] = field(default_factory=lambda: ["waste_efficiency", "overflow_rate"])
    morl_adaptation_rate: float = 0.1
