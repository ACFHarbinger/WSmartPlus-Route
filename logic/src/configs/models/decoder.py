"""
Decoder Config module.

Attributes:
    DecoderConfig: Decoder architecture and decoding configuration.

Example:
    >>> from logic.src.configs.models.decoder import DecoderConfig
    >>> config = DecoderConfig()
    >>> print(config)
    DecoderConfig(type='attention', embed_dim=128, hidden_dim=512, n_layers=3, n_heads=8, normalization=NormalizationConfig(name='layer', eps=1e-05, learn_scale=False, learn_shift=True), activation=ActivationConfig(name='gelu', param=1.0, threshold=6.0, replacement_value=6.0, n_params=3, range=[0.125, 0.3333333333333333]), decoding=DecodingConfig(type='greedy', min_tokens=1), dropout=0.1, mask_logits=True, connection_type='residual', n_predictor_layers=None, tanh_clipping=10.0, hyper_expansion=4)
"""

from dataclasses import dataclass, field
from typing import Optional

from .activation_function import ActivationConfig
from .decoding import DecodingConfig
from .normalization import NormalizationConfig


@dataclass
class DecoderConfig:
    """Decoder architecture and decoding configuration.

    Attributes:
        type (str): Decoding method.
        embed_dim (int): Dimension of the embedding vectors.
        hidden_dim (int): Dimension of the hidden layers.
        n_layers (int): Number of layers.
        n_heads (int): Number of attention heads.
        normalization (NormalizationConfig): Normalization configuration.
        activation (ActivationConfig): Activation function configuration.
        decoding (DecodingConfig): Decoding configuration.
        dropout (float): Dropout rate.
        mask_logits (bool): Whether to mask logits.
        connection_type (str): Type of connection.
        n_predictor_layers (Optional[int]): Number of predictor layers.
        tanh_clipping (float): Tanh clipping value.
        hyper_expansion (int): Hyper expansion value.
    """

    type: str = "attention"
    embed_dim: int = 128
    hidden_dim: int = 512
    n_layers: int = 3
    n_heads: int = 8
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    dropout: float = 0.1
    mask_logits: bool = True
    connection_type: str = "residual"
    n_predictor_layers: Optional[int] = None
    tanh_clipping: float = 10.0
    hyper_expansion: int = 4
