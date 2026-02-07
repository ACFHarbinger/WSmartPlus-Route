"""
DACT (Dual Aspect Collaborative Transformer) Policy implementation.

Reference:
    Ma et al. "Learning to Iteratively Solve Routing Problems with
    Dual-Aspect Collaborative Transformer" (2021)
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
    from logic.src.models.modules import MultiHeadAttention, Normalization
    from logic.src.models.modules.positional_embeddings import pos_init_embedding
else:
    from logic.src.envs.base import RL4COEnvBase
    from logic.src.models.modules import MultiHeadAttention, Normalization
    from logic.src.models.modules.positional_embeddings import pos_init_embedding


class DACTEncoder(ImprovementEncoder):
    """
    DACT Encoder: Combines node features (spatial) and solution features (positional).

    The "Dual Aspect" involves:
    1. Node Aspect: How nodes are located (Graph Convolution/Attention)
    2. Solution Aspect: Where nodes are in the current tour (Positional Encoding)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        pos_type: str = "CPE",
        **kwargs,
    ):
        """Initialize DACTEncoder."""
        super().__init__(embed_dim)

        # 1. Node aspect: init embedding + layers
        self.node_init = nn.Linear(2, embed_dim)

        # 2. Solution aspect: positional embedding
        self.pos_embedding = pos_init_embedding(pos_type, embed_dim)

        # 3. Collaborative Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mha": MultiHeadAttention(num_heads, embed_dim, embed_dim),
                        "norm1": Normalization(embed_dim, "layer"),
                        "ff": nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim)
                        ),
                        "norm2": Normalization(embed_dim, "layer"),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, td: TensorDict, **kwargs) -> torch.Tensor:
        """
        Encode problem instance and current solution.

        Args:
            td: TensorDict with 'locs' (customers), 'depot', and 'solution' (tour).

        Returns:
            Embeddings [batch, num_nodes, embed_dim].
        """
        # Combine depot and locs for full nodes
        depot = td["depot"].unsqueeze(1) if td["depot"].dim() == 2 else td["depot"]
        locs = td["locs"]
        nodes = torch.cat([depot, locs], dim=1)

        # Initial node embeddings
        h = self.node_init(nodes)

        # Apply positional encoding based on current solution order
        # solution: [batch, num_nodes] which is the tour (e.g., [0, 5, 2, ...])
        solution = td["solution"]
        num_nodes = solution.size(1)

        # Compute normalized positions for Periodic/Cyclic PE
        # Each node's position in the solution
        # (positions variable removed)

        # solution contains the node ID at each position
        # We want to find the position index p for each node ID n
        # This is essentially argsort of the solution
        _, pos_idx = solution.sort(dim=1)
        pos_normalized = pos_idx.float() / num_nodes

        # Apply PE
        h = self.pos_embedding(h, pos_normalized)

        # Pass through transformer layers
        for layer in self.layers:
            # Multi-head attention
            h_attn = layer["mha"](h)
            h = layer["norm1"](h + h_attn)

            # Feed forward
            h_ff = layer["ff"](h)
            h = layer["norm2"](h + h_ff)

        return h


class DACTDecoder(ImprovementDecoder):
    """
    DACT Decoder: Predicts 2-opt moves using cross-attention.

    A 2-opt move is defined by two indices (i, j) in the solution.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 8, **kwargs):
        """Initialize DACTDecoder."""
        super().__init__(embed_dim)
        self.num_heads = num_heads

        # Query/Key/Value projections for selecting i and j
        self.project_q = nn.Linear(embed_dim, embed_dim)
        self.project_k = nn.Linear(embed_dim, embed_dim)

        # Score normalization
        self.scale = 1.0 / (embed_dim // num_heads) ** 0.5

    def forward(
        self,
        td: TensorDict,
        embeddings: torch.Tensor | Tuple[torch.Tensor, ...],
        env: RL4COEnvBase,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict two nodes for 2-opt move.

        Returns:
            Tuple of (log_p, actions) where actions is [batch, 2].
        """
        if isinstance(embeddings, tuple):
            h = embeddings[0]
        else:
            h = embeddings
        bs, n, _ = h.shape

        # 1. Project to queries and keys
        q = self.project_q(h)  # [bs, n, d]
        k = self.project_k(h)  # [bs, n, d]

        # scores: [bs, n, n]
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale

        # 3. Mask invalid moves (i >= j for simplicity, or restricted moves)
        # For 2-opt in DACT, we typically pick i in [0, n-1] and j in [0, n-1]
        # But we must have i != j
        mask = torch.eye(n, device=h.device).bool().unsqueeze(0).expand(bs, -1, -1)
        scores = scores.masked_fill(mask, float("-inf"))

        # Flatten for softmax
        logits = scores.view(bs, -1)  # [bs, n*n]

        # 4. Sample action
        decode_type = kwargs.get("decode_type", "greedy")
        if decode_type == "greedy":
            action_indices = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action_indices = torch.multinomial(probs, 1).squeeze(-1)

        # Convert flattened index back to (i, j)
        idx_i = action_indices // n
        idx_j = action_indices % n

        actions = torch.stack([idx_i, idx_j], dim=-1)

        # Log likelihood
        log_p = F.log_softmax(logits, dim=-1).gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        return log_p, actions


class DACTPolicy(ImprovementPolicy):
    """
    DACT Policy: Collaborative Transformer for iterative improvement.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        env_name: str = "tsp_kopt",
        **kwargs,
    ):
        """Initialize DACTPolicy."""
        encoder = DACTEncoder(embed_dim, num_layers, num_heads, **kwargs)
        decoder = DACTDecoder(embed_dim, num_heads, **kwargs)

        super().__init__(encoder=encoder, decoder=decoder, env_name=env_name, embed_dim=embed_dim, **kwargs)
