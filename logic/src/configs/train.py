from dataclasses import dataclass
from typing import Optional


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
    """

    n_epochs: int = 100
    batch_size: int = 256
    train_data_size: int = 100000
    val_data_size: int = 10000
    val_dataset: Optional[str] = None
    num_workers: int = 0
    precision: str = "16-mixed"  # "16-mixed", "bf16-mixed", "32-true"
    # NEW FIELDS:
    train_time: bool = False
    eval_time_days: int = 1
    accumulation_steps: int = 1
    enable_scaler: bool = False
    checkpoint_epochs: int = 1
    shrink_size: Optional[int] = None
    post_processing_epochs: int = 0
    lr_post_processing: float = 0.001
    efficiency_weight: float = 0.8
    overflow_weight: float = 0.2
    log_step: int = 50
    # Process control
    epoch_start: int = 0
    eval_only: bool = False
    checkpoint_encoder: bool = False
    load_path: Optional[str] = None
    resume: Optional[str] = None
    eval_batch_size: int = 256
    persistent_workers: bool = True
    pin_memory: bool = False
