"""Cache for precomputed encoder outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PrecomputedCache:
    """Cache for precomputed encoder outputs."""

    node_embeddings: torch.Tensor
    graph_context: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor
