"""
HRL Manager setup utilities.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from logic.src.models import GATLSTManager
from logic.src.utils.functions import torch_load_cpu
from logic.src.interfaces import ITraversable


def setup_hrl_manager(
    opts: Dict[str, Any],
    device: torch.device,
    configs: Optional[Dict[str, Any]] = None,
    policy: Optional[str] = None,
    base_path: Optional[str] = None,
    worker_model: Optional[nn.Module] = None,
) -> Optional[GATLSTManager]:
    """
    Initializes and loads the Manager model for Hierarchical RL.

    Args:
        opts: Options dictionary.
        device: Torch device.
        configs: Configuration dictionary.
        policy: Policy name.
        base_path: Base path for models.
        worker_model: Worker model instance for shared encoder.

    Returns:
        The initialized manager model, or None if not applicable.
    """
    if configs is None:
        configs = {}

    hrl_path: Optional[str] = None
    if opts.get("model_path") is not None and policy is not None:
        model_paths: Dict[str, str] = opts["model_path"]
        if policy in model_paths:
            hrl_path = model_paths[policy]
        else:
            base_policy: str = policy.split("_gamma")[0].split("_emp")[0]
            if base_policy in model_paths:
                hrl_path = model_paths[base_policy]
            else:
                for key in model_paths:
                    if policy.startswith(key):
                        hrl_path = model_paths[key]
                        break

    if configs.get("mrl_method") != "hrl":
        return None

    if hrl_path is None:
        return None

    if base_path is not None and not os.path.exists(hrl_path):
        hrl_path = os.path.join(base_path, hrl_path)

    if os.path.isfile(hrl_path):
        pass
    elif os.path.isdir(hrl_path):
        pt_files: List[str] = [f for f in os.listdir(hrl_path) if f.endswith(".pt") and "epoch" in f]
        if not pt_files:
            return None
        epoch: int = max(int(os.path.splitext(f)[0].split("-")[1]) for f in pt_files)
        hrl_path = os.path.join(hrl_path, f"epoch-{epoch}.pt")
    else:
        return None

    mrl_history: int = configs.get("mrl_history", opts.get("mrl_history", 10))
    gat_hidden: int = configs.get("gat_hidden", opts.get("gat_hidden", 128))
    lstm_hidden: int = configs.get("lstm_hidden", opts.get("lstm_hidden", 64))
    global_input_dim: int = configs.get("global_input_dim", opts.get("global_input_dim", 3))

    load_data: Any = torch_load_cpu(hrl_path)
    state_dict: Dict[str, torch.Tensor] = (
        load_data["manager"] if isinstance(load_data, dict) and "manager" in load_data else load_data
    )

    if "gate_head.0.weight" in state_dict:
        weight_shape: torch.Size = state_dict["gate_head.0.weight"].shape
        in_dim: int = weight_shape[1]
        detected_dim: int = in_dim - gat_hidden
        if detected_dim > 0 and detected_dim != global_input_dim:
            global_input_dim = detected_dim

    manager: GATLSTManager = GATLSTManager(
        input_dim_static=2,
        input_dim_dynamic=mrl_history,
        hidden_dim=gat_hidden,
        lstm_hidden=lstm_hidden,
        device=device,
        global_input_dim=global_input_dim,
        shared_encoder=(
            worker_model.embedder
            if (worker_model is not None and opts.get("shared_encoder", True) and hasattr(worker_model, "embedder"))
            else None  # type: ignore
        ),
    ).to(device)

    manager.load_state_dict(state_dict)
    manager.eval()
    return manager
