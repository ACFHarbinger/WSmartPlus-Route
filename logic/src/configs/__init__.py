"""
Configuration dataclasses for WSmart-Route.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .core import RLConfig
from .envs import DataConfig, EnvConfig, GraphConfig, ObjectiveConfig
from .models import DecoderConfig, DecodingConfig, EncoderConfig, ModelConfig, OptimConfig
from .other import MustGoConfig, PostProcessingConfig
from .tasks import EvalConfig, HPOConfig, MetaRLConfig, SimConfig, TrainConfig


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
        must_go: Must-go selection strategy configuration.
        post_processing: Route refinement configuration.
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
    must_go: MustGoConfig = field(default_factory=MustGoConfig)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    seed: int = 42
    device: str = "cuda"
    experiment_name: Optional[str] = None
    task: str = "train"
    wandb_mode: str = "offline"
    no_tensorboard: bool = False
    no_progress_bar: bool = False
    output_dir: str = "assets/model_weights"
    log_dir: str = "logs"
    run_name: Optional[str] = None
    verbose: bool = True
    start: int = 0
    callbacks: Dict[str, Any] = field(default_factory=dict)


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
    "MustGoConfig",
    "PostProcessingConfig",
    "Config",
    "EncoderConfig",
    "DecoderConfig",
    "DecodingConfig",
    "GraphConfig",
    "ObjectiveConfig",
]
