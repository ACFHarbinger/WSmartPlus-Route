"""
This module contains the Deep Decoder Attention Model architecture.
"""

import math
import typing

import torch
import torch.nn as nn

from . import AttentionModel
from .subnets import GraphAttentionDecoder


class DeepAttentionModelFixed(typing.NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    mha_key: torch.Tensor

    def __getitem__(self, key):
        """
        Slice the fixed data.

        Args:
            key (slice or torch.Tensor): The slicing key.

        Returns:
            DeepAttentionModelFixed: A new instance with sliced data.
        """
        if torch.is_tensor(key) or isinstance(key, slice):
            return DeepAttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                mha_key=self.mha_key[key],
            )
        return self[key]


class DeepDecoderAttentionModel(AttentionModel):
    """
    Attention Model with a Deep Decoder architecture.

    Extends the standard AttentionModel to support a deeper decoder structure with
    more advanced configuration options (Graph Attention Decoder).
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        encoder_class,
        n_encode_layers=2,
        n_encode_sublayers=None,
        n_decode_layers=2,
        dropout_rate=0.1,
        aggregation="sum",
        aggregation_graph="avg",
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        mask_graph=False,
        normalization="batch",
        norm_learn_affine=True,
        norm_track_stats=False,
        norm_eps_alpha=1e-05,
        norm_momentum_beta=0.1,
        lrnorm_k=1.0,
        gnorm_groups=3,
        activation_function="gelu",
        af_param=1.0,
        af_threshold=6.0,
        af_replacement_value=6.0,
        af_num_params=3,
        af_uniform_range=[0.125, 1 / 3],
        n_heads=8,
        checkpoint_encoder=False,
        shrink_size=None,
        temporal_horizon=0,
        predictor_layers=None,
    ):
        """
        Initialize the Deep Decoder Attention Model.

        Args:
            embedding_dim (int): Dimension of the embedding vectors.
            hidden_dim (int): Dimension of the hidden layers.
            problem (object): The problem instance wrapper.
            encoder_class (object): The class for the encoder.
            n_encode_layers (int, optional): Number of encoder layers. Defaults to 2.
            n_encode_sublayers (int, optional): Number of sub-layers int encoder. Defaults to None.
            n_decode_layers (int, optional): Number of decoder layers. Defaults to 2.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            aggregation (str, optional): Aggregation method. Defaults to "sum".
            aggregation_graph (str, optional): Graph aggregation method. Defaults to "avg".
            tanh_clipping (float, optional): Tanh clipping value. Defaults to 10.0.
            mask_inner (bool, optional): Whether to mask inner attention. Defaults to True.
            mask_logits (bool, optional): Whether to mask logits. Defaults to True.
            mask_graph (bool, optional): Whether to mask graph attention. Defaults to False.
            normalization (str, optional): Normalization type. Defaults to 'batch'.
            norm_learn_affine (bool, optional): Learn affine parameters. Defaults to True.
            norm_track_stats (bool, optional): Track running stats. Defaults to False.
            norm_eps_alpha (float, optional): Epsilon/Alpha for norm. Defaults to 1e-05.
            norm_momentum_beta (float, optional): Momentum/Beta for norm. Defaults to 0.1.
            lrnorm_k (float, optional): K parameter for Local Response Norm. Defaults to 1.0.
            gnorm_groups (int, optional): Groups for Group Norm. Defaults to 3.
            activation_function (str, optional): Activation function name. Defaults to 'gelu'.
            af_param (float, optional): Parameter for activation function. Defaults to 1.0.
            af_threshold (float, optional): Threshold for activation function. Defaults to 6.0.
            af_replacement_value (float, optional): Replacement value for activation function. Defaults to 6.0.
            af_num_params (int, optional): Number of parameters for activation function. Defaults to 3.
            af_uniform_range (list, optional): Uniform range for activation params. Defaults to [0.125, 1/3].
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            checkpoint_encoder (bool, optional): Whether to checkpoint encoder. Defaults to False.
            shrink_size (int, optional): Shrink size. Defaults to None.
            temporal_horizon (int, optional): Temporal horizon. Defaults to 0.
            predictor_layers (int, optional): Number of layers in predictor. Defaults to None.
        """
        super(DeepDecoderAttentionModel, self).__init__(
            embedding_dim,
            hidden_dim,
            problem,
            encoder_class,
            n_encode_layers,
            n_encode_sublayers,
            dropout_rate,
            aggregation,
            aggregation_graph,
            tanh_clipping,
            mask_inner,
            mask_logits,
            mask_graph,
            normalization,
            norm_learn_affine,
            norm_track_stats,
            norm_eps_alpha,
            norm_momentum_beta,
            lrnorm_k,
            gnorm_groups,
            activation_function,
            af_param,
            af_threshold,
            af_replacement_value,
            af_num_params,
            af_uniform_range,
            n_heads,
            checkpoint_encoder,
            shrink_size,
            temporal_horizon,
        )
        self.n_decode_layers = n_decode_layers
        self.decoder = GraphAttentionDecoder(
            n_heads=self.n_heads,
            embed_dim=self.embedding_dim,
            n_layers=self.n_decode_layers,
            feed_forward_hidden=self.hidden_dim,
            normalization=self.normalization,
            epsilon_alpha=self.epsilon_alpha,
            learn_affine=self.learn_affine,
            track_stats=self.track_stats,
            momentum_beta=self.momentum_beta,
            locresp_k=self.locresp_k,
            n_groups=self.n_groups,
            activation=self.activation,
            af_param=self.af_param,
            threshold=self.threshold,
            replacement_value=self.replacement_value,
            n_params=self.n_params,
            uniform_range=self.uniform_range,
            dropout_rate=self.dropout_rate,
        )
        self.project_node_embeddings = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def _precompute(self, embeddings, num_steps=1):
        """
        Precompute fixed decoder context and keys.

        Args:
            embeddings (torch.Tensor): Node embeddings.
            num_steps (int, optional): Number of parallel steps. Defaults to 1.

        Returns:
            DeepAttentionModelFixed: Cached fixed context object.
        """
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        if self.aggregation_graph == "avg":
            graph_embed = embeddings.mean(1)
        elif self.aggregation_graph == "sum":
            graph_embed = embeddings.sum(1)
        elif self.aggregation_graph == "max":
            graph_embed = embeddings.max(1)[0]
        else:  # Default: disable graph embedding
            graph_embed = embeddings.sum(1) * 0.0

        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        mha_key = self.project_node_embeddings(embeddings[:, None, :, :])

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = self._make_heads(mha_key, num_steps)
        return DeepAttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p(self, fixed, state, normalize=True):
        """
        Compute log probabilities for the next step.

        Args:
            fixed (DeepAttentionModelFixed): Precomputed fixed context.
            state (object): Current decoder state.
            normalize (bool, optional): Whether to normalize logits. Defaults to True.

        Returns:
            tuple: (log_p, mask)
        """
        # Compute query = context node embedding
        query = fixed.context_node_projected + self.project_step_context(
            self._get_parallel_step_context(fixed.node_embeddings, state)
        )

        # Compute keys for the nodes
        mha_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        graph_mask = None
        if self.mask_graph:
            # Compute the graph mask, for masking next action based on graph structure
            graph_mask = state.get_edges_mask()

        # Compute logits (unnormalized log_p)
        log_p = self._one_to_many_logits(query, mha_K, mask, graph_mask)
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()
        return log_p, mask

    def _one_to_many_logits(self, query, mha_K, mask, graph_mask):
        """
        Compute logits for the one-to-many attention mechanism.

        Args:
            query (torch.Tensor): Query tensor.
            mha_K (torch.Tensor): Multi-head attention keys.
            mask (torch.Tensor): Valid action mask.
            graph_mask (torch.Tensor): Graph structure mask.

        Returns:
            torch.Tensor: Logits.
        """
        logits = self.decoder(query, mha_K, graph_mask)

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.mask_logits and self.mask_graph:
            logits[graph_mask] = -math.inf
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf
        return logits

    def _get_attention_node_data(self, fixed, state):
        """
        Get attention data for nodes, handling partial visits if needed.

        Args:
            fixed (DeepAttentionModelFixed): Fixed context.
            state (object): Current state.

        Returns:
            torch.Tensor: Attention keys.
        """
        if self.is_wc and self.allow_partial:
            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            mha_key_step = self.project_node_step(state.demands_with_depot[:, :, :, None].clone())

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return fixed.mha_key + self._make_heads(mha_key_step)

        return fixed.mha_key
