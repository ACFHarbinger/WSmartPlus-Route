"""MatNet Encoder.

Attributes:
    MatNetEncoder: MatNet Encoder that stacks multiple MatNetEncoderLayers.

Example:
    >>> from logic.src.models.subnets.encoders.matnet import MatNetEncoder
    >>> encoder = MatNetEncoder(num_layers=3, embed_dim=128, n_heads=8)
"""

from typing import Optional, Tuple

import torch
from torch import nn

from .matnet_encoder_layer import MatNetEncoderLayer


class MatNetEncoder(nn.Module):
    """MatNet Encoder.

    Stacks multiple MatNetEncoderLayer instances to process row and column
    embeddings for assignment problems.

    Attributes:
        layers (nn.ModuleList): List of MatNetEncoderLayer instances.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        n_heads: int,
        feed_forward_hidden: int = 512,
        normalization: str = "instance",
    ) -> None:
        """Initializes the MatNetEncoder.

        Args:
            num_layers: Number of encoder layers.
            embed_dim: Embedding dimension.
            n_heads: Number of attention heads.
            feed_forward_hidden: Hidden dimension for feed-forward layers.
            normalization: Type of normalization ("instance", "layer", or "batch").
        """
        super(MatNetEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [MatNetEncoderLayer(embed_dim, n_heads, feed_forward_hidden, normalization) for _ in range(num_layers)]
        )

    def forward(
        self,
        row_emb: torch.Tensor,
        col_emb: torch.Tensor,
        matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            row_emb: Row embeddings of shape (batch, row_size, embed_dim).
            col_emb: Column embeddings of shape (batch, col_size, embed_dim).
            matrix: Input cost/distance matrix of shape (batch, row_size, col_size).
            mask: Optional mask tensor of shape (batch, row_size, col_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated row and column embeddings.
        """
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, matrix, mask)
        return row_emb, col_emb
