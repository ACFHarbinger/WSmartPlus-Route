"""
Configuration and data sanitization utilities.
"""

from typing import Any

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


def get_pol_name(pol_obj: Any) -> str:
    """
    Extract a concise policy name from a structured or string config.
    Handles Hydra/OmegaConf objects, dicts, and strings.
    """
    try:
        # Sanitize to plain Python types (handles DictConfig, Tensor, etc.)
        sanitized = deep_sanitize(pol_obj)

        if isinstance(sanitized, dict):
            if len(sanitized) == 1:
                # Return the top-level key (e.g., 'rl_alns')
                return str(list(sanitized.keys())[0])
            return "complex_policy"

        if isinstance(sanitized, list):
            return f"policy_list_{len(sanitized)}"

        # If it's a string, or became one, ensure it's not a serialized dict/block
        name = str(sanitized)
        if "{" in name or "[" in name:
            import ast

            try:
                # Attempt to parse a string repr of a dict/list
                parsed = ast.literal_eval(name)
                return get_pol_name(parsed)
            except (ValueError, SyntaxError):
                return "unnamed_policy"

        if len(name) > 64:
            return "unnamed_policy"

        return name
    except Exception:
        return "unknown_policy"
