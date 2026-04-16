"""
Train Config module.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from ..envs.graph import GraphConfig
from ..envs.objective import ObjectiveConfig
from ..models.decoding import DecodingConfig
from ..policies.na import NeuralAgentConfig


@dataclass
class TrainConfig:
    """Training configuration.

    Attributes:
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        train_data_size: Number of training samples per epoch.
        val_data_size: Number of validation samples.
        val_dataset: Path to pre-generated validation dataset.
        num_workers: Number of data loading workers.
        data_distribution: Distribution for on-the-fly data generation.
        graph: Graph configuration.
        reward: Objective/reward configuration.
        decoding: Decoding configuration.
        model: Model configuration.
    """

    n_epochs: int = 100
    batch_size: int = 256
    train_data_size: int = 100000
    val_data_size: int = 10000
    val_dataset: Optional[str] = None
    num_workers: int = 4
    data_distribution: Optional[str] = None
    seed: int = 42
    precision: str = "16-mixed"  # "16-mixed", "bf16-mixed", "32-true"
    # NEW FIELDS:
    train_time: bool = False
    eval_time_days: int = 1
    accumulation_steps: int = 1
    enable_scaler: bool = False
    checkpoint_epochs: int = 1
    shrink_size: Optional[int] = None
    route_improvement_epochs: int = 0
    lr_route_improvement: float = 0.001
    efficiency_weight: float = 0.8
    overflow_weight: float = 0.2
    # Process control
    epoch_start: int = 0
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

    graph: GraphConfig = field(default_factory=GraphConfig)
    reward: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    policy: NeuralAgentConfig = field(default_factory=NeuralAgentConfig)
    callbacks: Optional[List[Any]] = None
