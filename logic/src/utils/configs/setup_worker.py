"""
Model loading and setup utilities.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from logic.src.utils.functions import load_model


def setup_model(
    policy: str,
    general_path: str,
    model_paths: Dict[str, str],
    device: torch.device,
    lock: threading.Lock,
    temperature: float = 1.0,
    decode_type: str = "greedy",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Sets up and loads a specific model based on policy.

    Args:
        policy: Policy identifier.
        general_path: Base path for models.
        model_paths: Mapping of policy names to file paths.
        device: Torch device.
        lock: Threading lock for safe loading.
        temperature: Softmax temperature. Defaults to 1.
        decode_type: Decoding strategy. Defaults to "greedy".

    Returns:
        tuple: (model, configs)
    """

    def _load_model(
        general_path: str,
        model_name: str,
        device: torch.device,
        temperature: float,
        decode_type: str,
        lock: threading.Lock,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        # Robust path handling: only join if model_name does not exist on its own
        if os.path.isabs(model_name) or os.path.exists(model_name):
            model_path = model_name
        else:
            model_path = os.path.join(general_path, model_name)

        if not os.path.exists(model_path):
            # Try joining with ROOT_DIR as fallback if it's a semi-absolute path like 'assets/...'
            from logic.src.constants import ROOT_DIR

            root_joined = os.path.join(ROOT_DIR, model_name)
            if os.path.exists(root_joined):
                model_path = root_joined

        with lock:
            model, configs = load_model(model_path)

        model.to(device)
        model.eval()
        if hasattr(model, "set_decode_type"):
            model.set_decode_type(decode_type, temp=temperature)
        return model, configs

    pol_strip: str = policy.rsplit("_", 1)[0]
    model_name = model_paths.get(pol_strip)

    if model_name is None:
        # Robust lookup: Try to find a key in model_paths that is exactly matching or a subset of the policy string
        # e.g. key 'amgat' should match 'means_std0.84_neural_amgat_gamma1'
        # Sort keys by length (desc) to find the most specific match first
        for key in sorted(model_paths.keys(), key=len, reverse=True):
            if key in policy:
                model_name = model_paths[key]
                break

    if model_name is None:
        raise KeyError(
            f"Could not find model path for policy '{policy}'. "
            f"Available keys in model_paths: {list(model_paths.keys())}"
        )

    return _load_model(general_path, model_name, device, temperature, decode_type, lock)
