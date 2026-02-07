from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from logic.src.constants.models import NODE_DIM

from .base import ContextEmbedder


class GenericContextEmbedder(ContextEmbedder):
    """
    Generic context embedder for problems without specific implementations.
    Embeds depot (2D) and nodes (node_dim) simply.
    """

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0):
        super().__init__(embed_dim, node_dim, temporal_horizon)
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)

    def init_node_embeddings(self, nodes: dict[str, Any]) -> torch.Tensor:
        # Fallback to 'loc' or 'locs'
        locs_key = "locs" if "locs" in nodes.keys() else "loc"
        node_features = nodes[locs_key]

        return torch.cat(
            (
                self.init_embed_depot(nodes["depot"])[:, None, :],
                self.init_embed(node_features),
            ),
            1,
        )

    @property
    def step_context_dim(self) -> int:
        return self.embed_dim
