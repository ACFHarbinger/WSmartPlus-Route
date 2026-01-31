"""
Configuration dataclasses for WSmart-Route.
"""

from dataclasses import dataclass, field
from typing import Optional

from .data import DataConfig
from .decoding import DecodingConfig
from .env import EnvConfig
from .eval import EvalConfig
from .hpo import HPOConfig
from .meta_rl import MetaRLConfig
from .model import ModelConfig
from .optim import OptimConfig
from .rl import RLConfig
from .sim import SimConfig
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
        eval: Evaluation configuration.
        sim: Simulation configuration.
        data: Data generation configuration.
        seed: Random seed.
        device: Device to use ('cpu', 'cuda').
        experiment_name: Optional name for the experiment.
        task: Task to perform ('train', 'eval', 'test_sim', 'gen_data').
    """

    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    meta_rl: MetaRLConfig = field(default_factory=MetaRLConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    device: str = "cuda"
    experiment_name: Optional[str] = None
    task: str = "train"
    # NEW FIELDS:
    wandb_mode: str = "offline"
    no_tensorboard: bool = False
    no_progress_bar: bool = False
    output_dir: str = "assets/model_weights"
    log_dir: str = "logs"
    run_name: Optional[str] = None
    verbose: bool = True
    start: int = 0


__all__ = [
    "EnvConfig",
    "ModelConfig",
    "TrainConfig",
    "OptimConfig",
    "RLConfig",
    "MetaRLConfig",
    "HPOConfig",
    "EvalConfig",
    "SimConfig",
    "DataConfig",
    "Config",
    "DecodingConfig",
]
