"""Graph Attention Convolution Encoder.

Attributes:
    GraphAttConvEncoder: Graph Attention Convolution Encoder using stacked AttentionConvolutionLayers.

Example:
    >>> from logic.src.models.subnets.encoders.gac import GraphAttConvEncoder
    >>> encoder = GraphAttConvEncoder(n_heads=8, embed_dim=128, n_layers=3)
"""

from __future__ import annotations

from typing import Any, List, Optional

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import TransformerEncoderBase

from .attention_convolution_layer import AttentionConvolutionLayer


class GraphAttConvEncoder(TransformerEncoderBase):
    """Graph Attention Convolution Encoder using stacked AttentionConvolutionLayers.

    Inherits from TransformerEncoderBase for common encoder patterns.

    Attributes:
        aggregate (str): Aggregation method for graph convolutions.
        n_sublayers (Optional[int]): Number of sublayers (unused).
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
        uniform_range: Optional[List[float]] = None,
        dropout_rate: float = 0.1,
        aggregate: str = "sum",
        **kwargs: Any,
    ) -> None:
        """Initializes the GraphAttConvEncoder.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            n_layers: Number of encoder layers.
            n_sublayers: Number of sublayers (unused, kept for API compatibility).
            feed_forward_hidden: Hidden dimension for feed-forward layers.
            normalization: Normalization type ("batch", "layer", "instance", or "group").
            epsilon_alpha: Epsilon for normalization stability.
            learn_affine: Whether to learn affine parameters in normalization.
            track_stats: Whether to track running stats in batch norm.
            momentum_beta: Momentum for batch norm running stats.
            locresp_k: Local response normalization parameter.
            n_groups: Number of groups for group normalization.
            activation: Activation function name.
            af_param: Activation function parameter.
            threshold: Activation threshold.
            replacement_value: Activation replacement value.
            n_params: Number of activation parameters.
            uniform_range: Uniform range for activation.
            dropout_rate: Dropout probability.
            aggregate: Aggregation method for graph convolutions.
            kwargs: Additional keyword arguments.
        """
        # Create config objects from individual parameters
        norm_config = NormalizationConfig(
            norm_type=normalization,
            epsilon=epsilon_alpha,
            learn_affine=learn_affine,
            track_stats=track_stats,
            momentum=momentum_beta,
            k_lrnorm=locresp_k,
            n_groups=n_groups,
        )

        activation_config = ActivationConfig(
            name=activation,
            param=af_param,
            threshold=threshold,
            replacement_value=replacement_value,
            n_params=n_params,
            range=uniform_range if uniform_range is not None else [0.125, 1 / 3],
        )

        # Store GAC-specific parameters BEFORE super().__init__
        # because the base class constructor calls _create_layer which needs these.
        self.aggregate = aggregate
        self.n_sublayers = n_sublayers

        # Initialize base class
        super(GraphAttConvEncoder, self).__init__(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            feed_forward_hidden=feed_forward_hidden,
            norm_config=norm_config,
            activation_config=activation_config,
            dropout_rate=dropout_rate,
            **kwargs,
        )

    def _create_layer(self, layer_idx: int) -> nn.Module:
        """Creates a single AttentionConvolutionLayer.

        Args:
            layer_idx: Index of the layer being created (0 to n_layers-1).

        Returns:
            nn.Module: AttentionConvolutionLayer instance.
        """
        return AttentionConvolutionLayer(
            self.n_heads,
            self.embed_dim,
            self.feed_forward_hidden,
            self.aggregate,
            self.norm_config.norm_type,
            self.norm_config.epsilon,
            self.norm_config.learn_affine,
            self.norm_config.track_stats,
            self.norm_config.momentum,
            self.norm_config.k_lrnorm,
            self.norm_config.n_groups,
            self.activation_config.name,
            self.activation_config.param,
            self.activation_config.threshold,
            self.activation_config.replacement_value,
            self.activation_config.n_params,
            self.activation_config.range,
        )
