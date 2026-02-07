"""
Main AttentionModel class assembling all mixins.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch.nn as nn

from logic.src.constants.models import (
    FEED_FORWARD_EXPANSION,
    NORM_EPSILON,
    TANH_CLIPPING,
)
from logic.src.models.subnets.factories import NeuralComponentFactory
from logic.src.utils.functions.problem import is_tsp_problem

from .decoding import DecodingMixin
from .forward import ForwardMixin
from .setup import SetupMixin


class AttentionModel(SetupMixin, ForwardMixin, DecodingMixin, nn.Module):
    """
    Attention Model for Vehicle Routing Problems.

    This model uses an Encoder-Decoder architecture with Multi-Head Attention to solve
    various VRP instances (VRPP, WCVRP, CWCVRP). It encodes the problem graph and
    constructively decodes the solution one step at a time.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        component_factory: NeuralComponentFactory,
        n_encode_layers: int = 2,
        n_encode_sublayers: Optional[int] = None,
        n_decode_layers: Optional[int] = None,
        dropout_rate: float = 0.1,
        aggregation: str = "sum",
        aggregation_graph: str = "avg",
        tanh_clipping: float = TANH_CLIPPING,
        mask_inner: bool = True,
        mask_logits: bool = True,
        mask_graph: bool = False,
        normalization: str = "batch",
        norm_learn_affine: bool = True,
        norm_track_stats: bool = False,
        norm_eps_alpha: float = NORM_EPSILON,
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
        hyper_expansion: int = FEED_FORWARD_EXPANSION,
        decoder_type: str = "attention",
    ) -> None:
        """
        Initialize the Attention Model.

        Args:
            embed_dim: Dimension of the embedding vectors.
            hidden_dim: Dimension of the hidden layers.
            problem: The problem instance wrapper (e.g., CVRRP, WCVRP).
            component_factory: Factory to create encoder/decoder components.
            n_encode_layers: Number of encoder layers. Defaults to 2.
            n_encode_sublayers: Number of sub-layers in each encoder layer.
            n_decode_layers: Number of decoder layers.
            dropout_rate: Dropout rate. Defaults to 0.1.
            aggregation: Aggregation block type. Defaults to "sum".
            aggregation_graph: Aggregation strategy for graph context ('avg', 'max', 'sum').
            tanh_clipping: Clipping value for tanh in attention. Defaults to 10.0.
            mask_inner: Whether to mask inner attention. Defaults to True.
            mask_logits: Whether to mask logits. Defaults to True.
            mask_graph: Whether to apply mask during graph encoding. Defaults to False.
            normalization: Normalization type ('batch', 'layer', 'instance', 'group').
            norm_learn_affine: Whether normalization learns affine parameters.
            norm_track_stats: Whether normalization tracks running ops.
            norm_eps_alpha: Epsilon/Alpha for normalization.
            norm_momentum_beta: Momentum/Beta for normalization.
            lrnorm_k: K parameter for LayerResponseNorm.
            gnorm_groups: Number of groups for GroupNorm.
            activation_function: Activation function name.
            af_param: Parameter for activation function (e.g., slope for LeakyReLU).
            af_threshold: Threshold for activation function.
            af_replacement_value: Replacement value for activation function.
            af_num_params: Number of learnable parameters for activation.
            af_uniform_range: Uniform range for initializing activation parameters.
            n_heads: Number of attention heads. Defaults to 8.
            checkpoint_encoder: Whether to checkpoint the encoder to save memory.
            shrink_size: Size to shrink the graph to (for large graphs).
            pomo_size: Number of starting nodes for POMO. Defaults to 0.
            temporal_horizon: Temporal horizon for time-dependent problems.
            spatial_bias: Whether to use spatial bias in attention.
            spatial_bias_scale: Scale factor for spatial bias.
            entropy_weight: Weight for entropy regularization optimization.
            predictor_layers: Number of layers in the predictor.
            connection_type: Type of skip connection ('residual', 'dense', 'highway').
            hyper_expansion: Expansion factor for feed-forward layers.
            decoder_type: Type of decoder ('attention', 'pointer', 'transformer').
        """
        nn.Module.__init__(self)
        SetupMixin.__init__(self)
        ForwardMixin.__init__(self)
        DecodingMixin.__init__(self)

        self._init_parameters(
            embed_dim,
            hidden_dim,
            problem,
            n_heads,
            pomo_size,
            checkpoint_encoder,
            aggregation_graph,
            temporal_horizon,
            tanh_clipping,
        )

        self._init_context_embedder(temporal_horizon)

        # Make step context dim dependent on problem/decoder
        step_context_dim = 2 * embed_dim + (embed_dim if is_tsp_problem(problem) else embed_dim)
        # Note: Logic for step_context_dim can be more complex depending on the specific problem
        # and decoder implementation, but this serves as a baseline.

        self._init_components(
            component_factory,
            step_context_dim,
            n_encode_layers,
            n_encode_sublayers,
            n_decode_layers,
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
            dropout_rate,
            aggregation,
            hyper_expansion,
            connection_type,
            predictor_layers,
            tanh_clipping,
            mask_inner,
            mask_logits,
            mask_graph,
            shrink_size,
            spatial_bias,
            spatial_bias_scale,
            decoder_type,
        )

        self.set_decode_type("greedy")
