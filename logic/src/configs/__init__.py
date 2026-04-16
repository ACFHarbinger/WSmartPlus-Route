"""
Configuration dataclasses for WSmart-Route.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .envs import EnvConfig, GraphConfig, ObjectiveConfig
from .models import DecoderConfig, DecodingConfig, EncoderConfig, ModelConfig, OptimConfig
from .policies.other import MandatorySelectionConfig, RouteImprovingConfig
from .rl import RLConfig
from .tasks import DataConfig, EvalConfig, HPOConfig, MetaRLConfig, SimConfig, SimHPOConfig, TrainConfig
from .tracking import TrackingConfig


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
        tracking: Tracking backend configuration (WSTracker + optional MLflow).
        mandatory_selection: Mandatory nodes selection strategy configuration.
        route_improvement: Route refinement configuration.
        load_dataset: Optional path to a dataset file to load.
        seed: Random seed.
        device: Device to use ('cpu', 'cuda').
        experiment_name: Optional name for the experiment.
        task: Task to perform ('train', 'eval', 'test_sim', 'gen_data').
        wandb_mode: Weights & Biases mode ('online', 'offline', 'disabled').
        no_tensorboard: If True, disable TensorBoard logging.
        no_progress_bar: If True, disable the progress bar.
        output_dir: Directory to save model outputs and artifacts.
        log_dir: Directory to save logs.
        run_name: Specific name for the run (separate from experiment_name).
        verbose: If True, enable verbose logging.
        start: Starting index or offset (e.g., for resuming or dataset slicing).
        p: Dictionary for arbitrary additional parameters.
        callbacks: Dictionary of callback configurations.
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
    hpo_sim: SimHPOConfig = field(default_factory=SimHPOConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mandatory_selection: MandatorySelectionConfig = field(default_factory=MandatorySelectionConfig)
    route_improvement: RouteImprovingConfig = field(default_factory=RouteImprovingConfig)
    load_dataset: Optional[str] = None
    seed: int = 42
    device: str = "cuda"
    experiment_name: Optional[str] = None
    task: str = "train"
    output_dir: str = "assets/model_weights"
    run_name: Optional[str] = None
    start: int = 0
    p: Dict[str, Any] = field(default_factory=dict)
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
    "SimHPOConfig",
    "DataConfig",
    "TrackingConfig",
    "MandatorySelectionConfig",
    "RouteImprovingConfig",
    "Config",
    "EncoderConfig",
    "DecoderConfig",
    "DecodingConfig",
    "GraphConfig",
    "ObjectiveConfig",
]
