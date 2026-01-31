"""
This module contains the Deep Decoder Attention Model architecture.
"""

import math
from typing import Any, List, NamedTuple, Optional

import torch
import torch.nn as nn

from logic.src.models.model_factory import NeuralComponentFactory

from . import AttentionModel
from .subnets import GraphAttentionDecoder


class DeepAttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached.
    This class allows for efficient indexing of multiple Tensors at once.
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
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        component_factory: NeuralComponentFactory,
        n_encode_layers: int = 2,
        n_encode_sublayers: Optional[int] = None,
        n_decode_layers: int = 2,
        dropout_rate: float = 0.1,
        aggregation: str = "sum",
        aggregation_graph: str = "avg",
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        mask_graph: bool = False,
        normalization: str = "batch",
        norm_learn_affine: bool = True,
        norm_track_stats: bool = False,
        norm_eps_alpha: float = 1e-05,
        norm_momentum_beta: float = 0.1,
        lrnorm_k: float = 1.0,
        gnorm_groups: int = 3,
        activation_function: str = "gelu",
        af_param: float = 1.0,
        af_threshold: float = 6.0,
        af_replacement_value: float = 6.0,
        af_num_params: int = 3,
        af_uniform_range: List[float] = [0.125, 1 / 3],
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        shrink_size: Optional[int] = None,
        pomo_size: int = 0,
        temporal_horizon: int = 0,
        spatial_bias: bool = False,
        spatial_bias_scale: float = 1.0,
        entropy_weight: float = 0.0,
        predictor_layers: Optional[int] = None,
        connection_type: str = "residual",
        hyper_expansion: int = 4,
    ) -> None:
        """
        Initialize the Deep Decoder Attention Model.
        """
        super(AttentionModel, self).__init__()
        self._init_parameters(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            n_heads=n_heads,
            pomo_size=pomo_size,
            checkpoint_encoder=checkpoint_encoder,
            aggregation_graph=aggregation_graph,
            temporal_horizon=temporal_horizon,
            tanh_clipping=tanh_clipping,
        )

        step_context_dim = self._init_context_embedder(temporal_horizon)

        self._init_components(
            component_factory=component_factory,
            step_context_dim=step_context_dim,
            n_encode_layers=n_encode_layers,
            n_encode_sublayers=n_encode_sublayers,
            dropout_rate=dropout_rate,
            normalization=normalization,
            norm_eps_alpha=norm_eps_alpha,
            norm_learn_affine=norm_learn_affine,
            norm_track_stats=norm_track_stats,
            norm_momentum_beta=norm_momentum_beta,
            lrnorm_k=lrnorm_k,
            gnorm_groups=gnorm_groups,
            activation_function=activation_function,
            af_param=af_param,
            af_threshold=af_threshold,
            af_replacement_value=af_replacement_value,
            af_num_params=af_num_params,
            af_uniform_range=af_uniform_range,
            aggregation=aggregation,
            connection_type=connection_type,
            hyper_expansion=hyper_expansion,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            mask_graph=mask_graph,
            shrink_size=shrink_size,
            spatial_bias=spatial_bias,
            spatial_bias_scale=spatial_bias_scale,
        )
        self.n_decode_layers = n_decode_layers
        self.n_decode_layers = n_decode_layers
        self.decoder = GraphAttentionDecoder(
            n_heads=self.n_heads,
            embed_dim=self.embed_dim,
            n_layers=self.n_decode_layers,
            feed_forward_hidden=self.hidden_dim,
            normalization=normalization,
            epsilon_alpha=norm_eps_alpha,
            learn_affine=norm_learn_affine,
            track_stats=norm_track_stats,
            momentum_beta=norm_momentum_beta,
            locresp_k=lrnorm_k,
            n_groups=gnorm_groups,
            activation=activation_function,
            af_param=af_param,
            threshold=af_threshold,
            replacement_value=af_replacement_value,
            n_params=af_num_params,
            uniform_range=af_uniform_range,
            dropout_rate=dropout_rate,
        )
        self.project_node_embeddings = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

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
