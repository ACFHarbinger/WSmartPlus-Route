"""MDAM Graph Attention Encoder.

Attributes:
    MDAMGraphAttentionEncoder: Multi-Decoder Attention Model Graph Attention Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.mdam import MDAMGraphAttentionEncoder
    >>> encoder = MDAMGraphAttentionEncoder(num_heads=8, embed_dim=128, num_layers=3)
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.autoregressive.encoder import AutoregressiveEncoder
from logic.src.models.subnets.modules.mdam_attention import MultiHeadAttentionMDAM
from logic.src.models.subnets.modules.normalization import Normalization
from logic.src.models.subnets.modules.skip_connection import SkipConnection

from .mdam_attention_layer import MultiHeadAttentionLayer


class MDAMGraphAttentionEncoder(AutoregressiveEncoder):
    """MDAM Graph Attention Encoder.

    Attributes:
        init_embed (Optional[nn.Linear]): Initial projection layer.
        layers (nn.Sequential): Stack of standard attention layers.
        attention_layer (MultiHeadAttentionMDAM): Last attention layer returning attn/V.
        norm1 (Normalization): First normalization layer.
        projection (SkipConnection): Feed-forward projection sublayer.
        norm2 (Normalization): Second normalization layer.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        num_layers: int,
        node_dim: Optional[int] = None,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
    ) -> None:
        """Initializes the MDAMGraphAttentionEncoder.

        Args:
            num_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            num_layers: Number of encoder layers.
            node_dim: Input feature dimension.
            normalization: Type of normalization ("batch", "layer", "instance").
            feed_forward_hidden: Hidden dimension for feed-forward layers.
        """
        super().__init__()

        # Optional input projection
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        # Standard attention layers (all but last)
        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feed_forward_hidden=feed_forward_hidden,
                    normalization=normalization,
                )
                for _ in range(num_layers - 1)
            )
        )

        # Last layer uses MDAM-specific attention to return attn/V
        self.attention_layer = MultiHeadAttentionMDAM(
            embed_dim=embed_dim,
            n_heads=num_heads,
            last_one=True,
        )

        # Post-attention processing
        self.norm1 = Normalization(embed_dim, normalization)
        self.projection = SkipConnection(
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(
        self,
        td: TensorDict,
        x: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes node features.

        Args:
            td: TensorDict containing observation data.
            x: Input node features of shape (batch, num_nodes, node_dim).
            mask: Optional attention mask (currently unsupported).
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Included node embeddings, graph embedding, attention weights,
                attention values, and intermediate node embeddings.

        Raises:
            ValueError: If input `x` is missing.
            AssertionError: If `mask` is provided.
        """
        assert mask is None, "Mask not yet supported in MDAM encoder"

        # Use provided x or extract from td
        if x is None:
            raise ValueError("MDAMGraphAttentionEncoder requires input x or pre-embedded features.")

        # Optional input projection
        h_embed = self.init_embed(x) if self.init_embed is not None else x

        # Apply standard attention layers
        h_old = self.layers(h_embed)

        # Apply MDAM attention (returns out, attn, V)
        h_new, attn, V = self.attention_layer(h_old)

        # Residual connection and normalization
        h = h_new + h_old
        h = self.norm1(h)
        h = self.projection(h)
        h = self.norm2(h)

        # Graph-level embedding is mean of node embeddings
        graph_embed = h.mean(dim=1)

        return h, graph_embed, attn, V, h_old

    def change(
        self,
        attn: torch.Tensor,
        V: torch.Tensor,
        h_old: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-computes embeddings with masked attention.

        Args:
            attn: Previous attention weights of shape (num_heads, batch, n, n).
            V: Previous attention values of shape (num_heads, batch, n, val_dim).
            h_old: Previous node embeddings before final attention layer.
            mask: Adjacency/mask tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: New node embeddings and graph embedding.
        """
        num_heads, batch_size, graph_size, feat_size = V.size()

        # Apply mask to attention weights
        mask_float = mask.float().view(1, batch_size, 1, graph_size)
        mask_float = mask_float.repeat(num_heads, 1, graph_size, 1)
        attn_masked = mask_float * attn

        # Renormalize attention
        attn_sum = torch.sum(attn_masked, dim=-1, keepdim=True) + 1e-9
        attn_masked = attn_masked / attn_sum

        # Compute new attention output
        heads = torch.matmul(attn_masked, V)

        # Project and reshape
        h_new = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.attention_layer.n_heads * self.attention_layer.val_dim),
            self.attention_layer.W_out.view(-1, self.attention_layer.embed_dim),
        ).view(batch_size, graph_size, self.attention_layer.embed_dim)

        # Residual and normalization
        h = h_new + h_old
        h = self.norm1(h)
        h = self.projection(h)
        h = self.norm2(h)

        # Graph-level embedding
        graph_embed = h.mean(dim=1)

        return h, graph_embed
