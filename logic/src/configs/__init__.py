"""
Configuration dataclasses for WSmart-Route.
"""

from dataclasses import dataclass, field
from typing import Optional

from .env import EnvConfig
from .hpo import HPOConfig
from .meta_rl import MetaRLConfig
from .model import ModelConfig
from .optim import OptimConfig
from .rl import RLConfig
from .train import TrainConfig


@dataclass
class Config:
    """Root configuration.

    Attributes:
        env: Environment configuration.
        model: Model configuration.
        train: Training configuration.
        optim: Optimizer configuration.
        rl: RL algorithm configuration.
        meta_rl: Meta-RL configuration.
        hpo: HPO configuration.
        seed: Random seed.
        device: Device to use ('cpu', 'cuda').
        experiment_name: Optional name for the experiment.
    """

    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    meta_rl: MetaRLConfig = field(default_factory=MetaRLConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    seed: int = 42
    device: str = "cuda"
    experiment_name: Optional[str] = None
    # NEW FIELDS:
    wandb_mode: str = "offline"
    no_tensorboard: bool = False
    no_progress_bar: bool = False
    output_dir: str = "assets/model_weights"
    log_dir: str = "logs"
    run_name: Optional[str] = None


__all__ = [
    "EnvConfig",
    "ModelConfig",
    "TrainConfig",
    "OptimConfig",
    "RLConfig",
    "MetaRLConfig",
    "HPOConfig",
    "Config",
]
