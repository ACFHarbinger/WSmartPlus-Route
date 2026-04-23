"""Attention capturing and monitoring hooks for Multi-Head Attention layers.

This module provides specialized hooks to intercept and record attention weights
and masks from the model's attention heads. It is particularly useful for
visualizing routing decisions and path importance in attention-based solvers.

Attributes:
    add_attention_hooks: Utility to register attention monitoring hooks.

Example:
    >>> from logic.src.tracking.hooks.attention_hooks import add_attention_hooks
    >>> hook_data = add_attention_hooks(model)
    >>> _ = model(x)
    >>> weights = hook_data['weights']
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import nn


def add_attention_hooks(model_module: nn.Module) -> Dict[str, Any]:
    """Registers forward hooks on Multi-Head Attention layers to capture weights.

    This function traverses the model's layers and attaches hooks to sub-modules
    specifically containing attention mechanisms.

    Args:
        model_module: The model or module to instrument.

    Returns:
        Dict[str, Any]: Mapping containing 'weights', 'masks', and 'handles'.
    """
    graph_masks: List[torch.Tensor] = []
    attention_weights: List[torch.Tensor] = []

    def hook(module: nn.Module, input_tensors: Tuple[torch.Tensor, ...], output: Any) -> None:
        """Internal hook to capture attention weights and masks from last_attn.

        Args:
            module: The attention module being executed.
            input_tensors: Layer inputs.
            output: Layer outputs.
        """
        last_attn = getattr(module, "last_attn", None)
        if last_attn is not None:
            # last_attn is expected to be a tuple/list (e.g. from PointerAttention)
            # where index 0 is weights and index -1 (or 1) is mask
            try:
                graph_masks.append(last_attn[-1])
                attention_weights.append(last_attn[0])
            except (IndexError, TypeError):
                pass

    # Register hooks on all MHA layers
    hook_handles: List[Any] = []
    hook_data: Dict[str, Any] = {"weights": attention_weights, "masks": graph_masks, "handles": hook_handles}
    for layer in getattr(model_module, "layers", []):
        # Get the actual attention module (skip the SkipConnection wrapper), if layer has attention
        if not hasattr(layer, "att"):
            continue
        attention_module = layer.att.module

        # Register hook and store the handle
        hook_handle = attention_module.register_forward_hook(hook)
        hook_handles.append(hook_handle)

    return hook_data
