"""
Model configuration dataclasses.

Attributes:
    ActivationConfig: Activation function configuration.
    DecoderConfig: Decoder configuration.
    DecodingConfig: Decoding configuration.
    EncoderConfig: Encoder configuration.
    ModelConfig: Model configuration.
    NormalizationConfig: Normalization configuration.
    OptimConfig: Optim configuration.

Example:
    >>> from logic.src.configs.models import ActivationConfig, DecoderConfig, DecodingConfig, EncoderConfig, ModelConfig, NormalizationConfig, OptimConfig
    >>> activation_config = ActivationConfig()
    >>> print(activation_config)
    ActivationConfig(type='gelu')
    >>> decoder_config = DecoderConfig()
    >>> print(decoder_config)
    DecoderConfig(q_ff=256, out_proj=64, n_heads=2, n_layers=3)
    >>> decoding_config = DecodingConfig()
    >>> print(decoding_config)
    DecodingConfig(temperature=1.0, softmax_temperature=None, num_beam_search=1, greedy_decode=False)
    >>> encoder_config = EncoderConfig()
    >>> print(encoder_config)
    EncoderConfig(enc_ff=256, n_heads=2, n_layers=2)
    >>> model_config = ModelConfig()
    >>> print(model_config)
    ModelConfig(n_customers=20, n_nodes=50, n_depots=1, embedding_dim=64, node_dim=128, n_heads=2, n_layers=6, dropout=0.1, normalization='layer', residual=True)
    >>> normalization_config = NormalizationConfig()
    >>> print(normalization_config)
    NormalizationConfig(embedding_dim=128, use_layer_norm=True)
    >>> optim_config = OptimConfig()
    >>> print(optim_config)
    OptimConfig(type='AdamW', lr=0.0003, weight_decay=0.1, clip=1.0, num_epochs=2000, early_stopping_patience=50, warmup_percentage=0.05, label_smoothing=0.1)
"""

from .activation_function import ActivationConfig
from .decoder import DecoderConfig
from .decoding import DecodingConfig
from .encoder import EncoderConfig
from .model import ModelConfig
from .normalization import NormalizationConfig
from .optim import OptimConfig

__all__ = [
    "ActivationConfig",
    "DecodingConfig",
    "DecoderConfig",
    "EncoderConfig",
    "ModelConfig",
    "NormalizationConfig",
    "OptimConfig",
]
