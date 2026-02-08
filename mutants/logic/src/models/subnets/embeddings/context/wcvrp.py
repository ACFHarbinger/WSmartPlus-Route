"""WC specific context embedder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from logic.src.constants.models import NODE_DIM, WC_STEP_CONTEXT_OFFSET

from .base import ContextEmbedder


class WCVRPContextEmbedder(ContextEmbedder):
    """Context Embedder for Waste Collection Vehicle Routing Problems (WCVRP)."""

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0):
        super().__init__(embed_dim, node_dim, temporal_horizon)
        input_dim = node_dim
        if temporal_horizon > 0:
            input_dim += temporal_horizon

        self.init_embed = nn.Linear(input_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)

    def init_node_embeddings(self, nodes: dict[str, Any], temporal_features: bool = True) -> torch.Tensor:
        locs = nodes["locs"]
        demand = nodes["demand"][:, :, None]

        node_features = [locs, demand]
        if "prize" in nodes:
            node_features.append(nodes["prize"][:, :, None])

        # Add temporal features if available
        if temporal_features and self.temporal_horizon > 0 and "temporal_features" in nodes:
            node_features.append(nodes["temporal_features"])

        node_features = torch.cat(node_features, -1)

        return torch.cat(
            (
                self.init_embed_depot(nodes["depot"])[:, None, :],
                self.init_embed(node_features),
            ),
            1,
        )

    @property
    def step_context_dim(self) -> int:
        return WC_STEP_CONTEXT_OFFSET
