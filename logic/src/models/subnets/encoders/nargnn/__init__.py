"""__init__.py module.

Attributes:
    NARGNNEncoder: Non-Autoregressive Graph Neural Network Encoder.
    NARGNNNodeEncoder: Node-only NARGNN Encoder.
    EdgeHeatmapGenerator: Generates heatmap from edge features.
    GNNLayer: Graph Neural Network Layer.
    SimplifiedEdgeEmbedding: Simplified edge embedding module.
    SimplifiedGNNEncoder: Simplified Graph Neural Network Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.nargnn import NARGNNEncoder
    >>> encoder = NARGNNEncoder(n_layers=3, embed_dim=128)
"""

from .edge_embedding import SimplifiedEdgeEmbedding as SimplifiedEdgeEmbedding
from .edge_heatmap_generator import EdgeHeatmapGenerator as EdgeHeatmapGenerator
from .encoder import NARGNNEncoder as NARGNNEncoder
from .gnn_encoder import SimplifiedGNNEncoder as SimplifiedGNNEncoder
from .gnn_layer import GNNLayer as GNNLayer
from .node_encoder import NARGNNNodeEncoder as NARGNNNodeEncoder

__all__ = [
    "NARGNNEncoder",
    "NARGNNNodeEncoder",
    "EdgeHeatmapGenerator",
    "GNNLayer",
    "SimplifiedEdgeEmbedding",
    "SimplifiedGNNEncoder",
]
