"""VRPP specific context embedder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from logic.src.constants.models import NODE_DIM

from .embedder_base import ContextEmbedder


class VRPPContextEmbedder(ContextEmbedder):
    """Context Embedder for VRP with Profits (VRPP) families."""

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0):
        super().__init__(embed_dim, node_dim, temporal_horizon)
        input_dim = node_dim
        if temporal_horizon > 0:
            input_dim += temporal_horizon

        self.init_embed = nn.Linear(input_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)

    def init_node_embeddings(self, nodes: dict[str, Any], temporal_features: bool = True) -> torch.Tensor:
        # Initial node embedding: locs + waste (demand) + prize
        # Waste and prize are usually [batch_size, num_nodes]
        # locs is [batch_size, num_nodes, 2]

        locs = nodes["locs"]
        waste = nodes["waste"][:, :, None]
        prize = nodes["prize"][:, :, None]

        node_features = [locs, waste, prize]

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
        # VRPP context usually adds remaining length/capacity info
        return 1
