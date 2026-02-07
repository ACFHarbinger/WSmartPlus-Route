"""
Base utilities and cache for MDAM decoder.
"""

from dataclasses import dataclass
from typing import Optional

import torch


def _decode_probs(
    probs: torch.Tensor,
    mask: Optional[torch.Tensor],
    decode_type: str = "sampling",
) -> torch.Tensor:
    """
    Decode action from probability distribution.

    Args:
        probs: Probability distribution (batch, num_nodes).
        mask: Valid action mask (batch, num_nodes).
        decode_type: 'greedy' or 'sampling'.

    Returns:
        Selected actions (batch,).
    """
    if mask is not None:
        probs = probs.masked_fill(~mask, 0.0)
        # Renormalize
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

    if decode_type == "greedy":
        return probs.argmax(dim=-1)
    else:
        # Sampling
        probs = probs.clamp(min=1e-8)
        return torch.multinomial(probs, 1).squeeze(-1)


@dataclass
class PrecomputedCache:
    """Cache for precomputed encoder outputs."""

    node_embeddings: torch.Tensor
    graph_context: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor
