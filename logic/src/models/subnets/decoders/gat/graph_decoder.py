"""Graph Attention Decoder module.

This module provides the multi-layer graph attention mechanism used by the GAT decoder
to refine queries against node embeddings.

Attributes:
    GraphAttentionDecoder: Multi-layer attention mechanism for decoding.

Example:
    >>> from logic.src.models.subnets.decoders.gat.graph_decoder import GraphAttentionDecoder
    >>> decoder = GraphAttentionDecoder(n_heads=8, embed_dim=128, n_layers=3)
    >>> out = decoder(query, node_embeddings)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig

from .multi_head_attention_layer import MultiHeadAttentionLayer


class GraphAttentionDecoder(nn.Module):
    """Decoder composed of stacked MultiHeadAttentionLayers.

    This module projects the refined query through multiple attention layers
    to compute selection probabilities.

    Attributes:
        layers (nn.ModuleList): Stacked MultiHeadAttentionLayer instances.
        projection (nn.Linear): Final projection layer to compute logits.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        feed_forward_hidden: int = 512,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        dropout_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initializes the GraphAttentionDecoder.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            n_layers: Number of attention layers.
            feed_forward_hidden: Hidden dimension for FFN.
            norm_config: Normalization configuration.
            activation_config: Activation configuration.
            dropout_rate: Dropout rate.
            kwargs: Additional keyword arguments.
        """
        super(GraphAttentionDecoder, self).__init__()

        if norm_config is None:
            norm_config = NormalizationConfig()
        if activation_config is None:
            activation_config = ActivationConfig()

        self.layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(
                    n_heads,
                    embed_dim,
                    feed_forward_hidden,
                    norm_config,
                    activation_config,
                )
                for _ in range(n_layers)
            ]
        )
        self.projection = nn.Linear(embed_dim, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        q: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the attention layers.

        Args:
            q: Query embeddings of shape (batch, ..., embed_dim).
            h: Node embeddings (keys/values). If None, uses q (self-attention).
            mask: Attention mask indicating valid nodes.

        Returns:
            torch.Tensor: Softmax probabilities over the sequence.
        """
        if h is None:
            h = q  # compute self-attention

        for layer in self.layers:
            h = layer(q, h, mask)

        out = self.projection(self.dropout(h))
        return torch.softmax(out, dim=-1)
