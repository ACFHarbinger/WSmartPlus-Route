"""
Graph Attention Decoder module.
"""

import torch
import torch.nn as nn

from .layers import MultiHeadAttentionLayer


class GraphAttentionDecoder(nn.Module):
    """
    Decoder composed of stacked MultiHeadAttentionLayers.
    Projects the final output to logits/probabilities.
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        feed_forward_hidden=512,
        normalization="batch",
        epsilon_alpha=1e-05,
        learn_affine=True,
        track_stats=False,
        momentum_beta=0.1,
        locresp_k=1.0,
        n_groups=3,
        activation="gelu",
        af_param=1.0,
        threshold=6.0,
        replacement_value=6.0,
        n_params=3,
        uniform_range=[0.125, 1 / 3],
        dropout_rate=0.1,
    ):
        """
        Initializes the GraphAttentionDecoder.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            n_layers: Number of layers.
            feed_forward_hidden: Hidden dimension of FFN.
            normalization: Type of normalization.
            epsilon_alpha: Epsilon value.
            learn_affine: Whether to learn affine parameters.
            track_stats: Whether to track stats.
            momentum_beta: Momentum value.
            locresp_k: K value for LocalResponseNorm.
            n_groups: Number of groups for GroupNorm.
            activation: Activation function.
            af_param: Activation parameter.
            threshold: Activation threshold.
            replacement_value: Replacement value.
            n_params: Number of parameters.
            uniform_range: Range for uniform distribution.
            dropout_rate: Dropout rate.
        """
        super(GraphAttentionDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(
                    n_heads,
                    embed_dim,
                    feed_forward_hidden,
                    normalization,
                    epsilon_alpha,
                    learn_affine,
                    track_stats,
                    momentum_beta,
                    locresp_k,
                    n_groups,
                    activation,
                    af_param,
                    threshold,
                    replacement_value,
                    n_params,
                    uniform_range,
                )
                for _ in range(n_layers)
            ]
        )
        self.projection = nn.Linear(embed_dim, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, h=None, mask=None):
        """
        Forward pass.

        Args:
            q: Query embeddings.
            h: Node embeddings (keys/values). If None, uses q (self-attention).
            mask: Attention mask.

        Returns:
            Softmax probabilities over the sequence.
        """
        if h is None:
            h = q  # compute self-attention

        for layer in self.layers:
            h = layer(q, h, mask)

        out = self.projection(self.dropout(h))
        return torch.softmax(out, dim=-1)
