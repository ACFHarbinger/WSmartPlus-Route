"""MoE Graph Attention Encoder.

Attributes:
    MoEGraphAttentionEncoder: MoE (Mixture of Experts) Graph Attention Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.moe import MoEGraphAttentionEncoder
    >>> encoder = MoEGraphAttentionEncoder(n_heads=8, embed_dim=128, n_layers=3)
"""

from __future__ import annotations

from typing import Any, List, Optional

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import TransformerEncoderBase

from .moe_multi_head_attention_layer import MoEMultiHeadAttentionLayer


class MoEGraphAttentionEncoder(TransformerEncoderBase):
    """MoE (Mixture of Experts) Graph Attention Encoder.

    Extends TransformerEncoderBase with MoE-specific multi-head attention layers.
    Each layer uses a mixture of expert feed-forward networks for improved
    capacity and specialization.

    Inherits from TransformerEncoderBase, which provides:
    - Layer stacking and management
    - Hyper-connection support (skip/dense/hyper)
    - Dropout application
    - Standard forward pass

    Attributes:
        num_experts (int): Number of expert networks in MoE.
        k (int): Number of experts to activate per input (top-k routing).
        noisy_gating (bool): Whether to use noisy gating for exploration.
        activation_function (str): Activation function name.
        af_param (float): Activation function parameter.
        af_threshold (float): Activation threshold.
        af_replacement_value (float): Activation replacement value.
        af_num_params (int): Number of activation parameters.
        af_uniform_range (List[float]): Uniform range for activation.
        normalization (str): Normalization type.
        norm_eps_alpha (float): Epsilon for normalization stability.
        norm_learn_affine (bool): Whether to learn affine parameters.
        norm_track_stats (bool): Whether to track running stats.
        norm_momentum_beta (float): Momentum for batch norm stats.
        lrnorm_k (float): Local response normalization parameter.
        gnorm_groups (int): Number of groups for group normalization.
        n_sublayers (Optional[int]): Number of sublayers (unused).
        agg (Optional[Any]): Aggregation method (unused).
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        n_sublayers: Optional[int] = None,
        feed_forward_hidden: int = 512,
        normalization: str = "batch",
        norm_eps_alpha: float = 1e-05,
        norm_learn_affine: bool = True,
        norm_track_stats: bool = False,
        norm_momentum_beta: float = 0.1,
        lrnorm_k: float = 1.0,
        gnorm_groups: int = 3,
        activation_function: str = "gelu",
        af_param: float = 1.0,
        af_threshold: float = 6.0,
        af_replacement_value: float = 6.0,
        af_num_params: int = 3,
        af_uniform_range: Optional[List[float]] = None,
        dropout_rate: float = 0.1,
        agg: Optional[Any] = None,
        connection_type: str = "skip",
        expansion_rate: int = 4,
        num_experts: int = 4,
        k: int = 2,
        noisy_gating: bool = True,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the MoEGraphAttentionEncoder.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            n_layers: Number of encoder layers.
            n_sublayers: Number of sublayers (unused, kept for API compatibility).
            feed_forward_hidden: Hidden dimension for each expert's feed-forward network.
            normalization: Normalization type ("batch", "layer", "instance", or "group").
            norm_eps_alpha: Epsilon for normalization stability.
            norm_learn_affine: Whether to learn affine parameters in normalization.
            norm_track_stats: Whether to track running stats in batch norm.
            norm_momentum_beta: Momentum for batch norm running stats.
            lrnorm_k: Local response normalization parameter.
            gnorm_groups: Number of groups for group normalization.
            activation_function: Activation function name.
            af_param: Activation function parameter.
            af_threshold: Activation threshold.
            af_replacement_value: Activation replacement value.
            af_num_params: Number of activation parameters.
            af_uniform_range: Uniform range for activation.
            dropout_rate: Dropout probability.
            agg: Aggregation method (unused, kept for API compatibility).
            connection_type: Connection type ("skip", "dense", or "hyper").
            expansion_rate: Expansion factor for hyper-connections.
            num_experts: Number of expert networks in MoE.
            k: Number of experts to activate per input (top-k routing).
            noisy_gating: Whether to use noisy gating for exploration.
            norm_config: Optional normalization configuration.
            activation_config: Optional activation function configuration.
            kwargs: Additional keyword arguments.
        """
        # Store MoE-specific parameters for _create_layer()
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        self.activation_function = activation_function
        self.af_param = af_param
        self.af_threshold = af_threshold
        self.af_replacement_value = af_replacement_value
        self.af_num_params = af_num_params
        self.af_uniform_range = af_uniform_range if af_uniform_range is not None else [0.125, 1 / 3]

        # Store parameters for layer creation
        self.normalization = normalization
        self.norm_eps_alpha = norm_eps_alpha
        self.norm_learn_affine = norm_learn_affine
        self.norm_track_stats = norm_track_stats
        self.norm_momentum_beta = norm_momentum_beta
        self.lrnorm_k = lrnorm_k
        self.gnorm_groups = gnorm_groups

        if norm_config is None:
            norm_config = NormalizationConfig(
                norm_type=normalization,
                epsilon=norm_eps_alpha,
                learn_affine=norm_learn_affine,
                track_stats=norm_track_stats,
                momentum=norm_momentum_beta,
                k_lrnorm=lrnorm_k,
                n_groups=gnorm_groups,
            )

        if activation_config is None:
            activation_config = ActivationConfig(
                name=activation_function,
                param=af_param,
                threshold=af_threshold,
                replacement_value=af_replacement_value,
                n_params=af_num_params,
                range=self.af_uniform_range,
            )

        # Initialize base class (handles layer creation, dropout, forward pass)
        super(MoEGraphAttentionEncoder, self).__init__(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            feed_forward_hidden=feed_forward_hidden,
            norm_config=norm_config,
            activation_config=activation_config,
            dropout_rate=dropout_rate,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
            **kwargs,
        )

        # Store unused parameters for potential future use
        self.n_sublayers = n_sublayers
        self.agg = agg

    def _create_layer(self, layer_idx: int) -> nn.Module:
        """Creates a single MoE multi-head attention layer.

        Args:
            layer_idx: Index of the layer being created (0 to n_layers-1).

        Returns:
            nn.Module: MoEMultiHeadAttentionLayer instance with MoE feed-forward.
        """
        return MoEMultiHeadAttentionLayer(
            self.n_heads,
            self.embed_dim,
            self.feed_forward_hidden,
            self.normalization,
            self.norm_eps_alpha,
            self.norm_learn_affine,
            self.norm_track_stats,
            self.norm_momentum_beta,
            self.lrnorm_k,
            self.gnorm_groups,
            self.activation_function,
            self.af_param,
            self.af_threshold,
            self.af_replacement_value,
            self.af_num_params,
            self.af_uniform_range,
            connection_type=self.conn_type,
            expansion_rate=self.expansion_rate,
            num_experts=self.num_experts,
            k=self.k,
            noisy_gating=self.noisy_gating,
        )
