from __future__ import annotations

import torch
import torch.nn as nn
from logic.src.models.subnets.modules import MultiHeadAttention, Normalization
from tensordict import TensorDict

from .common.improvement import ImprovementEncoder


class NeuOptEncoder(ImprovementEncoder):
    """
    NeuOpt Encoder: Transformer-based state encoding.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        **kwargs,
    ):
        """
        Initialize NeuOptEncoder.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            **kwargs: Unused arguments.
        """
        super().__init__(embed_dim=embed_dim)
        self.num_layers = num_layers

        # Initial projection for coordinates
        self.init_proj = nn.Linear(2, embed_dim)

        # Transformer layers
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

    def forward(self, td: TensorDict, **kwargs) -> torch.Tensor:
        """
        Encode graph state.
        """
        depot = td["depot"].unsqueeze(1)
        customers = td["locs"]
        h = torch.cat([depot, customers], dim=1)  # [bs, n, 2]

        h = self.init_proj(h)

        for layer in self.layers:
            # Multi-head attention
            attn_out = layer["mha"](h)
            h = layer["norm1"](h + attn_out)

            # Feed-forward
            ff_out = layer["ff"](h)
            h = layer["norm2"](h + ff_out)

        return h
