"""MoE Graph Attention Encoder."""

import torch.nn as nn
from logic.src.models.subnets.modules import MultiHeadAttention, Normalization
from logic.src.models.subnets.modules.connections import get_connection_module
from logic.src.models.subnets.modules.moe_feed_forward import MoEFeedForward


class MoEMultiHeadAttentionLayer(nn.Module):
    """
    Single layer of the MoE Graph Attention Encoder.
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        feed_forward_hidden,
        normalization,
        epsilon_alpha,
        learn_affine,
        track_stats,
        mbeta,
        lr_k,
        n_groups,
        activation,
        af_param,
        threshold,
        replacement_value,
        n_params,
        uniform_range,
        connection_type="skip",
        expansion_rate=4,
        num_experts=4,
        k=2,
        noisy_gating=True,
    ):
        """Initializes the MoEMultiHeadAttentionLayer."""
        super(MoEMultiHeadAttentionLayer, self).__init__()

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

        # Use MoEFeedForward instead of FeedForwardSubLayer
        self.ff = get_connection_module(
            module=MoEFeedForward(
                embed_dim,
                feed_forward_hidden,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
                num_experts=num_experts,
                k=k,
                noisy_gating=noisy_gating,
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

    def forward(self, h, mask=None):
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
            h_norm = self.norm1(h_perm)
            h = h_norm.permute(0, 1, 3, 2)
        else:
            h = self.norm1(h)

        h = self.ff(h)

        if h.dim() == 4:
            h_perm = h.permute(0, 1, 3, 2).contiguous()
            h_norm = self.norm2(h_perm)
            h = h_norm.permute(0, 1, 3, 2)
        else:
            return self.norm2(h)

        return h


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

        return self.dropout(curr)  # (batch_size, graph_size, embed_dim)
