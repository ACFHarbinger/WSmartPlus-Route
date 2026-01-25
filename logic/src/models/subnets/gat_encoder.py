"""Graph Attention Encoder."""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn

from ..modules import ActivationFunction, FeedForward, MultiHeadAttention, Normalization
from ..modules.connections import get_connection_module


class FeedForwardSubLayer(nn.Module):
    """
    Sub-layer containing a Feed-Forward Network and activation.
    """

    def __init__(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation: str,
        af_param: float,
        threshold: float,
        replacement_value: float,
        n_params: int,
        dist_range: List[float],
        bias: bool = True,
    ) -> None:
        """Initializes the FeedForwardSubLayer."""
        super(FeedForwardSubLayer, self).__init__()
        self.sub_layers = (
            nn.Sequential(
                FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                ActivationFunction(
                    activation,
                    af_param,
                    threshold,
                    replacement_value,
                    n_params,
                    (dist_range[0], dist_range[1]) if dist_range is not None and len(dist_range) >= 2 else None,
                ),
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
            if feed_forward_hidden > 0
            else FeedForward(embed_dim, embed_dim)
        )

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        return self.sub_layers(h)


class MultiHeadAttentionLayer(nn.Module):
    """
    Single layer of the Graph Attention Encoder.
    Uses connections factory for potential hyper-connections.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        normalization: str,
        epsilon_alpha: float,
        learn_affine: bool,
        track_stats: bool,
        mbeta: float,
        lr_k: float,
        n_groups: int,
        activation: str,
        af_param: float,
        threshold: float,
        replacement_value: float,
        n_params: int,
        uniform_range: List[float],
        connection_type: str = "skip",
        expansion_rate: int = 4,
    ) -> None:
        """Initializes the MultiHeadAttentionLayer."""
        super(MultiHeadAttentionLayer, self).__init__()

        self.att = get_connection_module(
            module=MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim),
            embed_dim=embed_dim,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

        self.norm1 = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )

        self.ff = get_connection_module(
            module=FeedForwardSubLayer(
                embed_dim,
                feed_forward_hidden,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
            ),
            embed_dim=embed_dim,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

        self.norm2 = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            h: Input features.
            mask: Attention mask.

        Returns:
            Updated features.
        """
        h = self.att(h, mask=mask)

        # Handle Norm for Hyper-Connections (4D input)
        if h.dim() == 4:  # (B, S, D, n)
            # Permute to (B, S, n, D) for Norm(D)
            h_perm = h.permute(0, 1, 3, 2).contiguous()
            h_norm: torch.Tensor = self.norm1(h_perm)
            h = h_norm.permute(0, 1, 3, 2)
        else:
            h = self.norm1(h)

        h = self.ff(h)

        if h.dim() == 4:
            h_perm = h.permute(0, 1, 3, 2).contiguous()
            h_norm = self.norm2(h_perm)
            h = h_norm.permute(0, 1, 3, 2)
        else:
            res: torch.Tensor = self.norm2(h)
            return res

        return h


class GraphAttentionEncoder(nn.Module):
    """
    Encoder composed of stacked MultiHeadAttentionLayers.
    Supports standard Transformer architecture and Hyper-Networks.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        n_sublayers: Optional[int] = None,
        feed_forward_hidden: int = 512,
        normalization: str = "batch",
        epsilon_alpha: float = 1e-05,
        learn_affine: bool = True,
        track_stats: bool = False,
        momentum_beta: float = 0.1,
        locresp_k: float = 1.0,
        n_groups: int = 3,
        activation: str = "gelu",
        af_param: float = 1.0,
        threshold: float = 6.0,
        replacement_value: float = 6.0,
        n_params: int = 3,
        uniform_range: List[float] = None,  # type: ignore
        dropout_rate: float = 0.1,
        agg: Any = None,
        connection_type: str = "skip",
        expansion_rate: int = 4,
        **kwargs,
    ) -> None:
        """
        Initializes the GraphAttentionEncoder.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            n_layers: Number of layers.
            n_sublayers: (Unused) Number of sublayers.
            feed_forward_hidden: Hidden dimension of FFN.
            normalization: Normalization type.
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
            agg: (Unused) Aggregation type.
            connection_type: Type of connection ('skip', 'static_hyper', etc.).
            expansion_rate: Expansion rate for hyper-connections.
        """
        super(GraphAttentionEncoder, self).__init__()

        # Set default uniform_range if None
        if uniform_range is None:
            uniform_range = [0.125, 1 / 3]

        self.conn_type = connection_type
        self.expansion_rate = expansion_rate

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
                    connection_type=connection_type,
                    expansion_rate=expansion_rate,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, edges: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (Batch, GraphSize, EmbedDim).
            edges: (Optional) Edge indices or mask.

        Returns:
            Encoded features (Batch, GraphSize, EmbedDim).
        """
        # 1. Expand Input (x -> H) if using Hyper-Connections
        if "hyper" in self.conn_type:
            # x: (B, S, D) -> H: (B, S, D, n)
            # Repeat input across all streams initially
            H = x.unsqueeze(-1).repeat(1, 1, 1, self.expansion_rate)
            curr = H
        else:
            curr = x

        # 2. Pass through layers
        for layer in self.layers:
            curr = layer(curr, mask=edges)

        # 3. Collapse Output (H -> x) if using Hyper-Connections
        if "hyper" in self.conn_type:
            # Simple mean pooling to return to standard embedding size
            curr = curr.mean(dim=-1)

        res: torch.Tensor = self.dropout(curr)
        return res  # (batch_size, graph_size, embed_dim)
