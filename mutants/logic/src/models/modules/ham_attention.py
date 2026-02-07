"""
HAM Attention Module.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from logic.src.models.modules.multi_head_attention import MultiHeadAttention
from logic.src.models.modules.normalization import Normalization


class HeterogeneousAttentionLayer(nn.Module):
    r"""
    Heterogeneous Multi-Head Attention Layer.

    Performs attention-based message passing between different node types.
    Can be configured to perform specific cross-attentions (e.g., Pickup->Delivery, Delivery->Pickup).

    Architecture:
    For each target type 't':
        h'_t = Norm(h_t + Agg(\sum_{s \in sources} Attention(q=h_t, k=h_s, v=h_s)))
        h''_t = Norm(h'_t + FFN(h'_t))
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],  # (source, relation, target)
        embed_dim: int,
        num_heads: int,
        feedforward_hidden: int = 512,
        normalization: str = "instance",
    ):
        """
        Initialize Heterogeneous Attention Layer.

        Args:
            node_types: List of node types.
            edge_types: List of edge triples (src, rel, dst).
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            feedforward_hidden: Hidden dimension.
            normalization: Normalization type.
        """
        """
        Initialize Heterogeneous Attention Layer.

        Args:
            node_types: List of node types.
            edge_types: List of edge triples (src, rel, dst).
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            feedforward_hidden: Hidden dimension.
            normalization: Normalization type.
        """
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.embed_dim = embed_dim

        # Creating MHA modules for each edge type
        # Key: (source, target) -> MHA
        # We assume relation name is just for semantics, but if we have multiple relations
        # between same pair, we'd need to handle that. Assuming unique pairs for now or unique keys.
        self.attn_modules = nn.ModuleDict()

        for src, rel, dst in edge_types:
            key = f"{src}_{rel}_{dst}"
            self.attn_modules[key] = MultiHeadAttention(n_heads=num_heads, input_dim=embed_dim, embed_dim=embed_dim)

        # Feed Forward per node type
        self.ffn = nn.ModuleDict(
            {
                nt: nn.Sequential(
                    nn.Linear(embed_dim, feedforward_hidden), nn.ReLU(), nn.Linear(feedforward_hidden, embed_dim)
                )
                for nt in node_types
            }
        )

        # Normalizations
        self.norm1 = nn.ModuleDict({nt: Normalization(embed_dim, normalization) for nt in node_types})
        self.norm2 = nn.ModuleDict({nt: Normalization(embed_dim, normalization) for nt in node_types})

    def forward(
        self, x_dict: Dict[str, torch.Tensor], mask_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: Dictionary of node features {node_type: (batch, num_nodes, embed_dim)}
            mask_dict: Not fully supported/needed for now, usually full attention.

        Returns:
            Updated x_dict.
        """

        # 1. Attention Aggregation
        # Accumulate updates for each target node type
        updates = {nt: torch.zeros_like(x_dict[nt]) for nt in self.node_types if nt in x_dict}
        counts = {nt: 0 for nt in self.node_types if nt in x_dict}

        for src, rel, dst in self.edge_types:
            if src not in x_dict or dst not in x_dict:
                continue

            key = f"{src}_{rel}_{dst}"
            mha = self.attn_modules[key]

            # Query from Dst, Key/Val from Src
            q = x_dict[dst]
            h = x_dict[src]

            # Simple MHA (Masking could be passed here if needed)
            out = mha(q=q, h=h, mask=None)

            updates[dst] = updates[dst] + out
            counts[dst] += 1

        # 2. Residual + Norm 1 + FFN + Norm 2
        out_dict = {}
        for nt in x_dict:
            x = x_dict[nt]

            # Apply attention update (mean aggregation or sum?)
            # Transformer implies we sum heads, but here we sum *edge types*.
            # Usually we sum.
            if counts[nt] > 0:
                # Residual
                x = x + updates[nt]

            x = self.norm1[nt](x)

            # FFN
            ff_out = self.ffn[nt](x)
            x = x + ff_out
            x = self.norm2[nt](x)

            out_dict[nt] = x

        return out_dict
