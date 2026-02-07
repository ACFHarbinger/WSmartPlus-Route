"""Graph Attention Encoder."""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn

from .gat_multi_head_attention_layer import GATMultiHeadAttentionLayer


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
        """
        super(GraphAttentionEncoder, self).__init__()

        # Set default uniform_range if None
        if uniform_range is None:
            uniform_range = [0.125, 1 / 3]

        self.conn_type = connection_type
        self.expansion_rate = expansion_rate

        self.layers = nn.ModuleList(
            [
                GATMultiHeadAttentionLayer(
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
            H = x.unsqueeze(-1).repeat(1, 1, 1, self.expansion_rate)
            curr = H
        else:
            curr = x

        # 2. Pass through layers
        for layer in self.layers:
            curr = layer(curr, mask=edges)

        # 3. Collapse Output (H -> x) if using Hyper-Connections
        if "hyper" in self.conn_type:
            curr = curr.mean(dim=-1)

        res: torch.Tensor = self.dropout(curr)
        return res  # (batch_size, graph_size, embed_dim)
