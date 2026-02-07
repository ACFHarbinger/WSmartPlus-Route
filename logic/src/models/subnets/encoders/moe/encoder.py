"""MoE Graph Attention Encoder."""

from __future__ import annotations

import torch.nn as nn

from .moe_multi_head_attention_layer import MoEMultiHeadAttentionLayer


class MoEGraphAttentionEncoder(nn.Module):
    """
    Encoder composed of stacked MoEMultiHeadAttentionLayers.
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        n_sublayers=None,
        feed_forward_hidden=512,
        normalization="batch",
        norm_eps_alpha=1e-05,
        norm_learn_affine=True,
        norm_track_stats=False,
        norm_momentum_beta=0.1,
        lrnorm_k=1.0,
        gnorm_groups=3,
        activation_function="gelu",
        af_param=1.0,
        af_threshold=6.0,
        af_replacement_value=6.0,
        af_num_params=3,
        af_uniform_range=[0.125, 1 / 3],
        dropout_rate=0.1,
        agg=None,
        connection_type="skip",
        expansion_rate=4,
        num_experts=4,
        k=2,
        noisy_gating=True,
        **kwargs,
    ):
        """
        Initializes the MoEGraphAttentionEncoder.
        """
        super(MoEGraphAttentionEncoder, self).__init__()

        self.conn_type = connection_type
        self.expansion_rate = expansion_rate

        self.layers = nn.ModuleList(
            [
                MoEMultiHeadAttentionLayer(
                    n_heads,
                    embed_dim,
                    feed_forward_hidden,
                    normalization,
                    norm_eps_alpha,
                    norm_learn_affine,
                    norm_track_stats,
                    norm_momentum_beta,
                    lrnorm_k,
                    gnorm_groups,
                    activation_function,
                    af_param,
                    af_threshold,
                    af_replacement_value,
                    af_num_params,
                    af_uniform_range,
                    connection_type=connection_type,
                    expansion_rate=expansion_rate,
                    num_experts=num_experts,
                    k=k,
                    noisy_gating=noisy_gating,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edges=None):
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

        return self.dropout(curr)  # (batch_size, graph_size, embed_dim)
