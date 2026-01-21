"""
Configuration dataclasses for WSmart-Route.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnvConfig:
    """Environment configuration."""

    name: str = "vrpp"
    num_loc: int = 50
    min_loc: float = 0.0
    max_loc: float = 1.0
    capacity: Optional[float] = None


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "am"
    embed_dim: int = 128
    hidden_dim: int = 512
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_heads: int = 8
    encoder_type: str = "gat"


@dataclass
class TrainConfig:
    """Training configuration."""

    n_epochs: int = 100
    batch_size: int = 256
    train_data_size: int = 100000
    val_data_size: int = 10000
    num_workers: int = 4


@dataclass
class OptimConfig:
    """Optimizer configuration."""

    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler: Optional[str] = None
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLConfig:
    """RL algorithm configuration."""

    algorithm: str = "reinforce"
    baseline: str = "rollout"
    entropy_weight: float = 0.0
    max_grad_norm: float = 1.0

    # PPO specific
    ppo_epochs: int = 10
    eps_clip: float = 0.2
    value_loss_weight: float = 0.5

    # SAPO specific
    sapo_tau_pos: float = 0.1
    sapo_tau_neg: float = 1.0

    # DR-GRPO specific
    dr_grpo_group_size: int = 8
    dr_grpo_epsilon: float = 0.2

    # Meta-RL specific
    use_meta: bool = False
    meta_lr: float = 1e-3
    meta_hidden_dim: int = 64
    meta_history_length: int = 10


@dataclass
class Config:
    """Root configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    seed: int = 42
    device: str = "cuda"
    experiment_name: Optional[str] = None


__all__ = [
    "EnvConfig",
    "ModelConfig",
    "TrainConfig",
    "OptimConfig",
    "RLConfig",
    "Config",
]
