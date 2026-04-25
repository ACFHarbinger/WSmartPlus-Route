"""
Encoder Config module.

Attributes:
    EncoderConfig: Encoder architecture configuration.

Example:
    >>> from logic.src.configs.models import EncoderConfig
    >>> config = EncoderConfig()
    >>> print(config)
    EncoderConfig(enc_ff=256, n_heads=2, n_layers=2)
"""

from dataclasses import dataclass, field
from typing import Optional

from .activation_function import ActivationConfig
from .normalization import NormalizationConfig


@dataclass
class EncoderConfig:
    """Encoder architecture configuration.

    Attributes:
        type (str): Type of encoder.
        embed_dim (int): Dimension of the embedding vectors.
        hidden_dim (int): Dimension of the hidden layers.
        n_layers (int): Number of layers.
        n_heads (int): Number of attention heads.
        n_sublayers (Optional[int]): Number of sublayers.
        normalization (NormalizationConfig): Normalization configuration.
        activation (ActivationConfig): Activation function configuration.
        dropout (float): Dropout rate.
        mask_inner (bool): Whether to mask inner layers.
        mask_graph (bool): Whether to mask graph.
        spatial_bias (bool): Whether to use spatial bias.
        connection_type (str): Type of connection.
        aggregation_graph (str): Aggregation method for graph.
        aggregation_node (str): Aggregation method for nodes.
        spatial_bias_scale (float): Scale of spatial bias.
        hyper_expansion (int): Hyper expansion value.
    """

    type: str = "gat"
    embed_dim: int = 128
    hidden_dim: int = 512
    n_layers: int = 3
    n_heads: int = 8
    n_sublayers: Optional[int] = None
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    dropout: float = 0.1
    mask_inner: bool = True
    mask_graph: bool = False
    spatial_bias: bool = False
    connection_type: str = "residual"

    # Aggregation params
    aggregation_graph: str = "avg"
    aggregation_node: str = "sum"

    # Other
    spatial_bias_scale: float = 1.0
    hyper_expansion: int = 4
