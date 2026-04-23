"""DACT Encoder implementation.

This module provides the `DACTEncoder`, which implements the "Dual Aspect"
encoding strategy. It processes the problem instance from two collaborative
perspectives: the spatial location of nodes (node aspect) and their relative
ordering in the current tour (solution aspect).

Attributes:
    DACTEncoder: Neural encoder for collaborative node-position modelling.

Example:
    >>> from logic.src.models.core.dact.encoder import DACTEncoder
    >>> encoder = DACTEncoder(embed_dim=128)
    >>> h = encoder(td)
"""

from __future__ import annotations

from typing import Any, cast

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.improvement.encoder import ImprovementEncoder
from logic.src.models.subnets.embeddings.positional import pos_init_embedding
from logic.src.models.subnets.modules import MultiHeadAttention, Normalization


class DACTEncoder(ImprovementEncoder):
    """DACT Dual Aspect Collaborative Encoder.

    Combines static node features with dynamic solution-dependent features
    using a Transformer backbone. The solution aspect uses normalized
    positional indices to represent the current tour order.

    Attributes:
        node_init (nn.Linear): Initial projection for spatial coordinates.
        pos_embedding (nn.Module): Positional encoding for tour sequence.
        layers (nn.ModuleList): Collaborative Transformer stacking.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        pos_type: str = "CPE",
        **kwargs: Any,
    ) -> None:
        """Initializes the DACT encoder.

        Args:
            embed_dim: Internal feature dimension.
            num_layers: count of transformation blocks.
            num_heads: count of attention heads per layer.
            pos_type: Identifier for the positional embedding scheme (e.g., 'CPE').
            **kwargs: Extra parameters.
        """
        super().__init__(embed_dim)

        self.node_init = nn.Linear(2, embed_dim)
        self.pos_embedding = pos_init_embedding(pos_type, embed_dim)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mha": MultiHeadAttention(num_heads, embed_dim, embed_dim),
                        "norm1": Normalization(embed_dim, "layer"),
                        "ff": nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 4),
                            nn.ReLU(),
                            nn.Linear(embed_dim * 4, embed_dim),
                        ),
                        "norm2": Normalization(embed_dim, "layer"),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, td: TensorDict, **kwargs: Any) -> torch.Tensor:
        """Encodes spatial nodes and their current tour positions.

        Args:
            td: state container with:
                - 'locs': [B, N, 2] coordinates.
                - 'depot': [B, 1, 2] or [B, 2] origin.
                - 'solution': [B, N+1] node indices representing the tour.
            **kwargs: unused.

        Returns:
            torch.Tensor: Refined node embeddings [B, N+1, embed_dim].
        """
        # Node processing
        depot = td["depot"].unsqueeze(1) if td["depot"].dim() == 2 else td["depot"]
        locs = td["locs"]
        nodes = torch.cat([depot, locs], dim=1)
        h = self.node_init(nodes)

        # Solution processing (Position Aspect)
        solution = td["solution"]
        num_nodes = solution.size(1)

        # Map node identity to sequence position
        _, pos_idx = solution.sort(dim=1)
        pos_normalized = pos_idx.float() / num_nodes

        # Inject positional info
        h = self.pos_embedding(h, pos_normalized)

        # Refine via Transformer
        for layer_module in self.layers:
            layer = cast(nn.ModuleDict, layer_module)
            h_attn = layer["mha"](h)
            h = layer["norm1"](h + h_attn)

            h_ff = layer["ff"](h)
            h = layer["norm2"](h + h_ff)

        return h
