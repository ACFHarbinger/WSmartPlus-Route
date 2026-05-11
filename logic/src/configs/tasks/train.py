"""
Train Config module.

Attributes:
    TrainConfig: Configuration for model training.

Example:
    train_config = TrainConfig(
        batch_size=256,
        val_dataset="val.pkl",
        num_workers=4,
        data_distribution="gaussian",
    )
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from logic.src.configs.envs.env import EnvConfig
from logic.src.configs.models.decoding import DecodingConfig
from logic.src.configs.policies.na import NeuralAgentConfig


@dataclass
class TrainConfig:
    """Training configuration.

    Attributes:
        batch_size: Training batch size.
        val_dataset: Path to pre-generated validation dataset.
        num_workers: Number of data loading workers.
        data_distribution: Distribution for on-the-fly data generation.
        env: Environment configuration (contains graph, reward, curriculum_graphs, eval_graphs).
        decoding: Decoding configuration.
    """

    batch_size: int = 256
    val_dataset: Optional[str] = None
    num_workers: int = 4
    data_distribution: Optional[str] = None
    seed: int = 42
    precision: str = "16-mixed"  # "16-mixed", "bf16-mixed", "32-true"
    # NEW FIELDS:
    train_time: bool = False
    accumulation_steps: int = 1
    enable_scaler: bool = False
    checkpoint_epochs: int = 1
    shrink_size: Optional[int] = None
    route_improvement_epochs: int = 0
    lr_route_improvement: float = 0.001
    efficiency_weight: float = 0.8
    overflow_weight: float = 0.2
    # Process control
    eval_only: bool = False
    checkpoint_encoder: bool = False
    resume: Optional[str] = None
    model_weights_path: Optional[str] = None
    final_model_path: Optional[str] = None
    eval_batch_size: int = 256
    persistent_workers: bool = True
    pin_memory: bool = False
    reload_dataloaders_every_n_epochs: int = 1
    devices: Union[int, str] = "auto"
    strategy: Optional[str] = "auto"

    env: Any = field(default_factory=EnvConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    policy: NeuralAgentConfig = field(default_factory=NeuralAgentConfig)
    callbacks: Optional[List[Any]] = None
