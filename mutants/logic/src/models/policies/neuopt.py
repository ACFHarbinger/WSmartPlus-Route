"""
Neural Optimizer (NeuOpt) Policy.

Guided iterative improvement for combinatorial optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from .common.improvement import (
    ImprovementDecoder,
    ImprovementEncoder,
    ImprovementPolicy,
)

if TYPE_CHECKING:
    from logic.src.envs.base import RL4COEnvBase
    from logic.src.models.subnets.modules import MultiHeadAttention, Normalization
else:
    from logic.src.models.subnets.modules import MultiHeadAttention, Normalization


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


class NeuOptDecoder(ImprovementDecoder):
    """
    NeuOpt Decoder: Guided move selection.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        super().__init__(embed_dim=embed_dim)
        self.project_q = nn.Linear(embed_dim, embed_dim)
        self.project_k = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5

    def forward(
        self,
        td: TensorDict,
        embeddings: torch.Tensor | Tuple[torch.Tensor, ...],
        env: RL4COEnvBase,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict move (idx1, idx2).
        """
        if isinstance(embeddings, tuple):
            h = embeddings[0]
        else:
            h = embeddings
        bs, n, _ = h.shape

        q = self.project_q(h)  # [bs, n, d]
        k = self.project_k(h)  # [bs, n, d]

        # scores: [bs, n, n]
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale

        # Mask invalid moves (diagonal)
        mask = torch.eye(n, device=h.device).bool().unsqueeze(0).expand(bs, -1, -1)
        scores.masked_fill_(mask, float("-inf"))

        logits = scores.view(bs, -1)

        decode_type = kwargs.get("decode_type", "greedy")
        if decode_type == "greedy":
            action_indices = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action_indices = torch.multinomial(probs, 1).squeeze(-1)

        # Convert flat index back to (i, j)
        idx1 = torch.div(action_indices, n, rounding_mode="floor")
        idx2 = action_indices % n
        actions = torch.stack([idx1, idx2], dim=-1)

        log_p = F.log_softmax(logits, dim=-1).gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        return log_p, actions


class NeuOptPolicy(ImprovementPolicy):
    """
    NeuOpt Policy: Iterative guided search.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__(env_name="tsp_kopt", embed_dim=embed_dim)
        self.encoder = NeuOptEncoder(embed_dim, num_heads, num_layers)
        self.decoder = NeuOptDecoder(embed_dim)
