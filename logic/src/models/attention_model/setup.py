"""
Setup logic for AttentionModel.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch.nn as nn

from logic.src.constants.models import NODE_DIM
from logic.src.models.subnets.embeddings import (
    ContextEmbedder,
    GenericContextEmbedder,
    VRPPContextEmbedder,
    WCVRPContextEmbedder,
)
from logic.src.models.subnets.factories import NeuralComponentFactory
from logic.src.utils.functions.problem import is_tsp_problem, is_vrpp_problem, is_wc_problem


class SetupMixin:
    """Mixin for initialization and component setup."""

    def __init__(self):
        # Type hints for attributes initialized here but used elsewhere
        self.encoder: nn.Module
        self.decoder: nn.Module
        self.context_embedder: ContextEmbedder
        self.problem: Any

    def _init_parameters(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        n_heads: int,
        pomo_size: int,
        checkpoint_encoder: bool,
        aggregation_graph: str,
        temporal_horizon: int,
        tanh_clipping: float,
    ):
        """Initialize basic model parameters."""
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.problem = problem
        self.pomo_size = pomo_size
        self.checkpoint_encoder = checkpoint_encoder
        self.aggregation_graph = aggregation_graph
        self.temporal_horizon = temporal_horizon
        self.tanh_clipping = tanh_clipping

    def _init_context_embedder(self, temporal_horizon: int):
        """Initialize the context embedder strategy."""
        if is_vrpp_problem(self.problem):
            self.context_embedder = VRPPContextEmbedder(self.embed_dim, temporal_horizon=self.temporal_horizon)
        elif is_wc_problem(self.problem):
            self.context_embedder = WCVRPContextEmbedder(self.embed_dim, temporal_horizon=self.temporal_horizon)
        else:
            node_dim = 2 if is_tsp_problem(self.problem) else NODE_DIM
            self.context_embedder = GenericContextEmbedder(
                self.embed_dim,
                node_dim=node_dim,
                temporal_horizon=self.temporal_horizon,
            )

    @property
    def is_vrpp(self):
        return is_vrpp_problem(self.problem)

    @property
    def is_wc(self):
        return is_wc_problem(self.problem)

    def _init_components(
        self,
        component_factory: NeuralComponentFactory,
        step_context_dim: int,
        n_encode_layers: int,
        n_encode_sublayers: Optional[int],
        n_decode_layers: Optional[int],
        normalization: str,
        norm_learn_affine: bool,
        norm_track_stats: bool,
        norm_eps_alpha: float,
        norm_momentum_beta: float,
        lrnorm_k: float,
        gnorm_groups: int,
        activation_function: str,
        af_param: float,
        af_threshold: float,
        af_replacement_value: float,
        af_num_params: int,
        af_uniform_range: List[float],
        dropout_rate: float,
        aggregation: str,
        hyper_expansion: int,
        connection_type: str,
        predictor_layers: Optional[int],
        tanh_clipping: float,
        mask_inner: bool,
        mask_logits: bool,
        mask_graph: bool,
        shrink_size: Optional[int],
        spatial_bias: bool,
        spatial_bias_scale: float,
        decoder_type: str = "attention",
    ):
        """Initialize encoder and decoder components using the factory."""
        # Encoder
        self.encoder = component_factory.create_encoder(
            embed_dim=self.embed_dim,
            n_layers=n_encode_layers,
            n_sublayers=n_encode_sublayers,
            normalization=normalization,
            norm_learn_affine=norm_learn_affine,
            norm_track_stats=norm_track_stats,
            norm_eps_alpha=norm_eps_alpha,
            norm_momentum_beta=norm_momentum_beta,
            lrnorm_k=lrnorm_k,
            gnorm_groups=gnorm_groups,
            activation_function=activation_function,
            af_param=af_param,
            af_threshold=af_threshold,
            af_replacement_value=af_replacement_value,
            af_num_params=af_num_params,
            af_uniform_range=af_uniform_range,
            dropout_rate=dropout_rate,
            aggregation=aggregation,
            hyper_expansion=hyper_expansion,
            connection_type=connection_type,
            n_heads=self.n_heads,
            mask_inner=mask_inner,
            mask_graph=mask_graph,
            spatial_bias=spatial_bias,
            spatial_bias_scale=spatial_bias_scale,
        )

        # Decoder
        self.decoder = component_factory.create_decoder(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            problem=self.problem,
            n_layers=n_decode_layers,
            n_heads=self.n_heads,
            step_context_dim=step_context_dim,
            predictor_layers=predictor_layers,
            normalization=normalization,
            norm_learn_affine=norm_learn_affine,
            norm_track_stats=norm_track_stats,
            norm_eps_alpha=norm_eps_alpha,
            norm_momentum_beta=norm_momentum_beta,
            lrnorm_k=lrnorm_k,
            gnorm_groups=gnorm_groups,
            activation_function=activation_function,
            af_param=af_param,
            af_threshold=af_threshold,
            af_replacement_value=af_replacement_value,
            af_num_params=af_num_params,
            af_uniform_range=af_uniform_range,
            dropout_rate=dropout_rate,
            aggregation=aggregation,
            hyper_expansion=hyper_expansion,
            connection_type=connection_type,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            shrink_size=shrink_size,
            decoder_type=decoder_type,
        )
