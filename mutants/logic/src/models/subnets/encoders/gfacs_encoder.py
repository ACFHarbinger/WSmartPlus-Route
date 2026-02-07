"""
GFACS Encoder.

GNN-based encoder with log-partition function estimation for training with Trajectory Balance loss.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from logic.src.models.subnets.encoders.nargnn_encoder import NARGNNEncoder
from tensordict.tensordict import TensorDict


class GFACSEncoder(NARGNNEncoder):
    """
    NARGNN Encoder with log-partition function estimation for training with Trajectory Balance loss.
    Trajectory Balance (TB) loss (Malkin et al., https://arxiv.org/abs/2201.13259).

    Extends NARGNNEncoder with a Z-network that estimates the log-partition
    function (logZ), which is used for computing the TB loss during training.

    Args:
        embed_dim: Embedding dimension.
        env_name: Environment name.
        init_embedding: Custom initial embedding module (optional).
        edge_embedding: Custom edge embedding module (optional).
        graph_network: Custom graph network module (optional).
        heatmap_generator: Custom heatmap generator module (optional).
        num_layers_heatmap_generator: Number of layers in heatmap generator.
        num_layers_graph_encoder: Number of layers in graph encoder.
        num_layers_Z_net: Number of layers in Z-network for logZ estimation.
        act_fn: Activation function for GNN layers.
        agg_fn: Aggregation function for GNN layers.
        linear_bias: Use bias in linear layers.
        **kwargs: Additional arguments passed to parent NARGNNEncoder.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        env_name: str = "tsp",
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        graph_network: Optional[nn.Module] = None,
        heatmap_generator: Optional[nn.Module] = None,
        num_layers_heatmap_generator: int = 5,
        num_layers_graph_encoder: int = 15,
        num_layers_Z_net: int = 3,
        act_fn: str = "silu",
        agg_fn: str = "mean",
        linear_bias: bool = True,
        **kwargs,
    ) -> None:
        """Initialize GFACS encoder."""
        super().__init__(
            embed_dim=embed_dim,
            env_name=env_name,
            init_embedding=init_embedding,
            edge_embedding=edge_embedding,
            graph_network=graph_network,
            heatmap_generator=heatmap_generator,
            num_layers_heatmap_generator=num_layers_heatmap_generator,
            num_layers_graph_encoder=num_layers_graph_encoder,
            act_fn=act_fn,
            agg_fn=agg_fn,
            linear_bias=linear_bias,
            **kwargs,
        )

        # Z-network for log-partition function estimation
        # Used in Trajectory Balance (TB) loss
        z_out_dim = kwargs.get("z_out_dim", 1)
        layers = []
        for _ in range(num_layers_Z_net - 1):
            layers.extend([nn.Linear(embed_dim, embed_dim), nn.ReLU()])
        layers.append(nn.Linear(embed_dim, z_out_dim))
        self.Z_net = nn.Sequential(*layers)

    def forward(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """
        Forward pass of the GFACS encoder.

        Note: This extends the parent NARGNNEncoder by adding log-partition
        function estimation (logZ), which is required for TB loss computation.
        The return type is intentionally different from parent (3-tuple vs 2-tuple).

        Args:
            td: TensorDict with problem data.

        Returns:
            Tuple of (heatmap_logits, node_embed, logZ) where:
                - heatmap_logits: Edge heatmap probabilities [batch, n, n]
                - node_embed: Initial node embeddings [batch, n, embed_dim]
                - logZ: Log-partition function estimates [batch, 1]
        """
        # Get heatmap and embeddings from parent
        heatmap_logits, node_embed = super().forward(td)

        # Compute log-partition function estimate
        # Average pool node embeddings and pass through Z-network
        graph_embedding = node_embed.mean(dim=1)  # [batch, embed_dim]
        logZ = self.Z_net(graph_embedding)  # [batch, 1]

        return heatmap_logits, node_embed, logZ
