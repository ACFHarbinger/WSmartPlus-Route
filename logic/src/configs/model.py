"""
Model Config module.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        name: Name of the model architecture (e.g., 'am', 'deep_decoder').
        embed_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        encoder_type: Type of encoder ('gat', 'gcn', etc.).
    """

    name: str = "am"
    embed_dim: int = 128
    hidden_dim: int = 512
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_heads: int = 8
    encoder_type: str = "gat"
    # NEW FIELDS:
    temporal_horizon: int = 0
    tanh_clipping: float = 10.0
    normalization: str = "instance"
    activation: str = "gelu"
    dropout: float = 0.1
    mask_inner: bool = True
    mask_logits: bool = True
    mask_graph: bool = False
    spatial_bias: bool = False
    connection_type: str = "residual"
    # Hyper-parameters for specialized layers
    num_encoder_sublayers: Optional[int] = None
    num_predictor_layers: Optional[int] = None
    learn_affine: bool = True
    track_stats: bool = False
    epsilon_alpha: float = 1e-5
    momentum_beta: float = 0.1
    lrnorm_k: Optional[float] = None
    gnorm_groups: int = 4
    activation_param: float = 1.0
    activation_threshold: Optional[float] = None
    activation_replacement: Optional[float] = None
    activation_num_parameters: int = 3
    activation_uniform_range: List[float] = field(default_factory=lambda: [0.125, 0.333])
    aggregation_graph: str = "avg"
    aggregation_node: str = "sum"
    spatial_bias_scale: float = 1.0
    hyper_expansion: int = 4
