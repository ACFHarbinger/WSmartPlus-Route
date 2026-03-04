"""
Configuration and data sanitization utilities.
"""

from typing import Any, Dict, Union

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def deep_sanitize(val: Any) -> Any:
    """
    Recursively convert NumPy types, Tensors, and OmegaConf containers to Python primitives.

    Args:
        val: The value to sanitize.

    Returns:
        The sanitized value (dict, list, int, float, str, etc.).
    """
    if val is None:
        return None

    # Handle OmegaConf containers
    if isinstance(val, (DictConfig, ListConfig)):
        val = OmegaConf.to_container(val, resolve=True)

    # Handle NumPy scalars
    if isinstance(val, np.generic):
        return val.item()

    # Handle NumPy arrays
    if isinstance(val, np.ndarray):
        return val.tolist()

    # Handle PyTorch Tensors
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        return val.detach().cpu().numpy().tolist()

    # Handle standard containers
    if isinstance(val, dict):
        return {str(k): deep_sanitize(v) for k, v in val.items()}

    if isinstance(val, (list, tuple)):
        return [deep_sanitize(item) for item in val]

    return val


def get_pol_name(pol_obj: Union[str, Dict[str, Any]]) -> str:
    """Extract policy name from structured or string config."""
    if isinstance(pol_obj, dict):
        if len(pol_obj) == 1:
            return list(pol_obj.keys())[0]
        return "complex_policy"
    return str(pol_obj)
