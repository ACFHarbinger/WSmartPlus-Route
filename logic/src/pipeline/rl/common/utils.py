"""
Common utilities for RL training.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.optim import Optimizer


def get_optimizer(
    name: str,
    parameters: Any,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> Optimizer:
    """
    Get optimizer by name.

    Args:
        name: Optimizer name ('adam', 'adamw', 'sgd', 'rmsprop').
        parameters: Model parameters to optimize.
        lr: Learning rate.
        weight_decay: Weight decay factor.
        **kwargs: Additional optimizer arguments.

    Returns:
        Initialized optimizer.
    """
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    **kwargs: Any,
) -> Optional[Any]:
    """
    Get learning rate scheduler by name.

    Args:
        name: Scheduler name ('cosine', 'step', 'exponential', 'plateau', 'lambda', 'none').
        optimizer: Optimizer instance.
        **kwargs: Scheduler arguments.

    Returns:
        Initialized scheduler or None.
    """
    name = name.lower()
    if name == "none":
        return None
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif name == "lambda":
        # Assumes kwargs contains 'lr_lambda' callable
        if "lr_lambda" not in kwargs:
            raise ValueError("LambdaLR requires 'lr_lambda' in kwargs")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def get_lightning_device(trainer: Any) -> torch.device:
    """
    Get the device from a PyTorch Lightning Trainer.

    Args:
        trainer: PL Trainer instance.

    Returns:
        torch.device: Determine device.
    """
    if trainer.accelerator == "gpu":
        return torch.device(f"cuda:{trainer.strategy.root_device.index}")
    elif trainer.accelerator == "cpu":
        return torch.device("cpu")
    # Add other accelerators if needed
    return torch.device("cpu")
