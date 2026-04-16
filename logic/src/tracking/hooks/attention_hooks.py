"""
Attention hooks and capturing utilities.
"""

from __future__ import annotations

from typing import Any, Dict

from torch import nn


def add_attention_hooks(model_module: nn.Module) -> Dict[str, Any]:
    """
    Registers forward hooks on Multi-Head Attention layers to capture weights and masks.

    Args:
        model_module: The model to hook.

    Returns:
        dict: Dictionary containing lists of 'weights', 'masks', and 'handles'.
    """
    graph_masks = []
    attention_weights = []

    def hook(module: nn.Module, input: Any, output: Any) -> None:
        """Forward hook to capture attention weights."""
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
    hook_data = {"weights": attention_weights, "masks": graph_masks, "handles": []}
    for layer in getattr(model_module, "layers", []):
        # Get the actual attention module (skip the SkipConnection wrapper), if layer has attention
        if not hasattr(layer, "att"):
            continue
        attention_module = layer.att.module

        # Register hook and store the handle
        hook_handle = attention_module.register_forward_hook(hook)
        hook_data["handles"].append(hook_handle)
    return hook_data
