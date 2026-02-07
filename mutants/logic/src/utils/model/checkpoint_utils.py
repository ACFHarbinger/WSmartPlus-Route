"""
Checkpoint loading utilities for PyTorch models.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def torch_load_cpu(load_path: str) -> Any:
    """
    Loads a checkpoint file mapping all tensors to CPU.

    Args:
        load_path: Path to the checkpoint file.

    Returns:
        The loaded data.
    """
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def load_data(load_path: Optional[str], resume: Optional[str]) -> Any:
    """
    Loads data from a path or resume checkpoint.

    Args:
        load_path: Explicit path to data.
        resume: Path to resume checkpoint.

    Returns:
        Loaded data or empty dict if neither is provided.
    """
    data = {}
    assert load_path is None or resume is None, "Only one of load path and resume can be given"

    load_path = load_path if load_path is not None else resume
    if load_path is not None:
        print("  [*] Loading data from {}".format(load_path))
        data = torch_load_cpu(load_path)
    return data


def _load_model_file(load_path: str, model: nn.Module) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
    """
    Loads the model with parameters from the file and returns optimizer state dict if it is in the file.

    Args:
        load_path: Path to the checkpoint.
        model: Model to load parameters into.

    Returns:
        Tuple of (model, optimizer state dict).
    """
    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print("  [*] Loading model from {}".format(load_path))

    load_data = torch.load(os.path.join(os.getcwd(), load_path), map_location=lambda storage, loc: storage)
    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get("optimizer", None)
        load_model_state_dict = load_data.get("model", load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)
    model.load_state_dict(state_dict)
    return model, load_optimizer_state_dict
