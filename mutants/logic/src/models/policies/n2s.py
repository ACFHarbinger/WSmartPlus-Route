"""
Neural Neighborhood Search (N2S) Policy.

A lightweight improvement model that uses neighborhood-constrained attention
to predict local search moves.
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
    from logic.src.envs.base import RL4COEnvBase
    from logic.src.models.subnets.modules import MultiHeadAttention, Normalization


class N2SEncoder(ImprovementEncoder):
    """
    N2S Encoder: A lightweight transformer with neighborhood masking.

    Attends only to K-nearest neighbors to improve efficiency and focus
    on local search moves.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        k_neighbors: int = 20,
        **kwargs,
    ):
        """
        Initialize N2SEncoder.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            k_neighbors: Number of nearest neighbors to attend to.
            **kwargs: Unused arguments.
        """
        super().__init__(embed_dim=embed_dim)
        self.k_neighbors = k_neighbors
        self.num_heads = num_heads

        self.mha = MultiHeadAttention(num_heads, embed_dim, embed_dim)
        self.norm = Normalization(embed_dim, "layer")
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def _get_neighborhood_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute mask for K-nearest neighbors based on spatial distance.
        """
        # locs: [bs, n, 2]
        depot = td["depot"].unsqueeze(1)
        customers = td["locs"]
        locs = torch.cat([depot, customers], dim=1)

        # dist: [bs, n, n]
        dist = torch.cdist(locs, locs)

        # Get indices of K-nearest neighbors
        # (self is always included in nearest)
        _, topk_indices = dist.topk(self.k_neighbors + 1, largest=False, sorted=False)

        # Create mask: True for excluded neighbors
        bs, n, _ = dist.shape
        mask = torch.ones(bs, n, n, device=dist.device, dtype=torch.bool)
        mask.scatter_(2, topk_indices, False)

        return mask

    def forward(self, td: TensorDict, **kwargs) -> torch.Tensor:
        """
        Encode nodes with neighborhood-constrained attention.
        """
        # Initial embedding (coords)
        depot = td["depot"].unsqueeze(1)
        customers = td["locs"]
        h = torch.cat([depot, customers], dim=1)  # [bs, n, 2]

        # Linear projection if needed (h is [bs, n, 2], we want embed_dim)
        # Assuming we have an init_embedding or simple linear
        if not hasattr(self, "init_proj"):
            self.init_proj = nn.Linear(2, self.embed_dim).to(h.device)

        h = self.init_proj(h)

        # Neighborhood mask
        mask = self._get_neighborhood_mask(td)

        # Transformer layer
        attn_out = self.mha(h, mask=mask)
        h = self.norm(h + attn_out)
        h = h + self.ff(h)

        return h


class N2SDecoder(ImprovementDecoder):
    """
    N2S Decoder: Predicts moves within node neighborhoods.
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

        # Mask invalid moves
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


class N2SPolicy(ImprovementPolicy):
    """
    N2S Policy: Iterative neighborhood search.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        k_neighbors: int = 20,
        **kwargs,
    ):
        super().__init__(env_name="tsp_kopt", embed_dim=embed_dim)
        self.encoder = N2SEncoder(embed_dim, num_heads, k_neighbors)
        self.decoder = N2SDecoder(embed_dim)
