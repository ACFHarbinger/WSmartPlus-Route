"""Graph Attention Decoder."""

import math
import typing

import torch
import torch.nn as nn

from logic.src.models.modules import (
    ActivationFunction,
    FeedForward,
    MultiHeadAttention,
    Normalization,
    SkipConnection,
)


class FeedForwardSubLayer(nn.Module):
    """
    Sub-layer containing a Feed-Forward Network and activation.
    """

    def __init__(
        self,
        embed_dim,
        feed_forward_hidden,
        activation,
        af_param,
        threshold,
        replacement_value,
        n_params,
        dist_range,
        bias=True,
    ):
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
                    dist_range,
                ),
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
            if feed_forward_hidden > 0
            else FeedForward(embed_dim, embed_dim)
        )

    def forward(self, h, mask=None):
        """Forward pass."""
        return self.sub_layers(h)


class MultiHeadAttentionLayer(nn.Module):
    """
    Single layer of the Graph Attention Decoder.
    Contains Multi-Head Attention followed by a Feed-Forward block, with normalization.
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
    ):
        """Initializes the MultiHeadAttentionLayer."""
        super(MultiHeadAttentionLayer, self).__init__()
        self.att = SkipConnection(MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim))
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
        self.ff = SkipConnection(
            FeedForwardSubLayer(
                embed_dim,
                feed_forward_hidden,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
            )
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

    def forward(self, q, h, mask):
        """Forward pass."""
        h = self.att(q, h, mask)
        h = self.norm1(h)
        h = self.ff(h)
        return self.norm2(h)


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


class DeepAttentionModelFixed(typing.NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor

    def __getitem__(self, key):
        """Get item from DeepAttentionModelFixed."""
        if torch.is_tensor(key) or isinstance(key, slice):
            return DeepAttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
            )
        return DeepAttentionModelFixed(
            node_embeddings=self.node_embeddings[key].unsqueeze(0),
            context_node_projected=self.context_node_projected[key].unsqueeze(0),
        )


class DeepGATDecoder(nn.Module):
    """
    Deep Decoder module independent of the model architecture.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        normalization: str = "batch",
        dropout_rate: float = 0.1,
        aggregation_graph: str = "avg",
        mask_graph: bool = False,
        mask_logits: bool = True,
        tanh_clipping: float = 10.0,
        temp: float = 1.0,
        **kwargs,
    ):
        """Initialize DeepGATDecoder."""
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.aggregation_graph = aggregation_graph
        self.mask_graph = mask_graph
        self.mask_logits = mask_logits
        self.tanh_clipping = tanh_clipping
        self.temp = temp

        self.decoder = GraphAttentionDecoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            feed_forward_hidden=hidden_dim,
            normalization=normalization,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        # Projections
        # Input to project_fixed_context is graph_embed (embed_dim)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_step_context = nn.Linear(2 * embed_dim + 1, embed_dim, bias=False)

    def _precompute(self, embeddings, num_steps=1):
        if self.aggregation_graph == "avg":
            graph_embed = embeddings.mean(1)
        elif self.aggregation_graph == "sum":
            graph_embed = embeddings.sum(1)
        elif self.aggregation_graph == "max":
            graph_embed = embeddings.max(1)[0]
        else:
            graph_embed = embeddings.sum(1) * 0.0

        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        return DeepAttentionModelFixed(embeddings, fixed_context)

    def _get_log_p(self, fixed, state, normalize=True):
        step_context = self._get_parallel_step_context(fixed.node_embeddings, state)

        query = fixed.context_node_projected + self.project_step_context(step_context)

        # Pass raw embeddings to decoder
        mha_K = fixed.node_embeddings

        mask = state.get_mask()
        graph_mask = state.get_edges_mask() if self.mask_graph else None

        log_p = self._one_to_many_logits(query, mha_K, mask, graph_mask)
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        return log_p, mask

    def _one_to_many_logits(self, query, mha_K, mask, graph_mask):
        # Interact with GATDecoder layers directly to refine query
        # Standard MHA: q_new = layer(q, k, mask)
        # Note: GATDecoder stores embeddings in mha_K.
        # GATDecoder.forward uses h=q for self attention if h is None.
        # But here we pass query and mha_K.

        q = query
        for layer in self.decoder.layers:
            q = layer(q, mha_K, mask)

        # Compute proper pointer attention logits: (B, 1, Nodes)
        # logits = (q @ mha_K^T) / sqrt(dim)
        logits = torch.matmul(q, mha_K.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        if self.mask_logits and self.mask_graph:
            logits[graph_mask] = -math.inf

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        if self.mask_logits:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            # Apply mask to logits
            logits[mask] = -math.inf

        return logits

    def _get_parallel_step_context(self, embeddings, state):
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if num_steps > 1:
            raise NotImplementedError("Parallel steps > 1 not supported yet")

        current_node_embed = torch.gather(
            embeddings,
            1,
            current_node.unsqueeze(-1).expand(-1, -1, embeddings.size(-1)),
        ).view(batch_size, num_steps, -1)

        return torch.cat(
            [
                current_node_embed,
                torch.zeros_like(current_node_embed),  # Placeholder for graph context/other
                torch.zeros(batch_size, num_steps, 1, device=embeddings.device),
            ],
            dim=-1,
        )
