"""
RL Config module.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PPOConfig:
    """PPO specific configuration."""

    epochs: int = 10
    eps_clip: float = 0.2
    value_loss_weight: float = 0.5
    mini_batch_size: float = 0.25


@dataclass
class SAPOConfig:
    """SAPO specific configuration."""

    tau_pos: float = 0.1
    tau_neg: float = 1.0


@dataclass
class GRPOConfig:
    """GRPO specific configuration."""

    group_size: int = 8
    epsilon: float = 0.2
    epochs: int = 3


@dataclass
class POMOConfig:
    """POMO specific configuration."""

    num_augment: int = 1
    num_starts: Optional[int] = None
    augment_fn: str = "dihedral8"


@dataclass
class SymNCOConfig:
    """SymNCO specific configuration."""

    alpha: float = 0.2
    beta: float = 1.0


@dataclass
class ImitationConfig:
    """Imitation specific configuration."""

    mode: str = "hgs"  # hgs, alns, random_ls, 2opt
    random_ls_iterations: int = 100
    random_ls_op_probs: Optional[Dict[str, float]] = None
    enabled: bool = False
    loss_fn: str = "nll"


@dataclass
class GDPOConfig:
    """GDPO specific configuration."""

    objective_keys: List[str] = field(default_factory=lambda: ["reward_prize", "reward_cost"])
    objective_weights: Optional[List[float]] = None
    conditional_key: Optional[str] = None
    renormalize: bool = True


@dataclass
class AdaptiveImitationConfig:
    """Adaptive Imitation Learning specific configuration."""

    il_weight: float = 1.0
    il_decay: float = 0.95
    patience: int = 5
    threshold: float = 0.05
    decay_step: int = 1
    epsilon: float = 1e-5


@dataclass
class RLConfig:
    """RL algorithm configuration.

    Attributes:
        algorithm: RL algorithm name ('reinforce', 'ppo', 'sapo', etc.).
        baseline: Baseline type ('rollout', 'critic', 'pomo', 'warmup').
        entropy_weight: Weight for entropy regularization.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    algorithm: str = "reinforce"
    baseline: str = "rollout"
    bl_warmup_epochs: int = 0
    entropy_weight: float = 0.0
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    exp_beta: float = 0.8
    bl_alpha: float = 0.05

    # Algorithm specific sub-configs
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sapo: SAPOConfig = field(default_factory=SAPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    pomo: POMOConfig = field(default_factory=POMOConfig)
    symnco: SymNCOConfig = field(default_factory=SymNCOConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    gdpo: GDPOConfig = field(default_factory=GDPOConfig)
    adaptive_imitation: AdaptiveImitationConfig = field(default_factory=AdaptiveImitationConfig)
