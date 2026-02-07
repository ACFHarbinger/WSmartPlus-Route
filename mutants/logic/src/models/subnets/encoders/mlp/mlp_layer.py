"""MLP layer implementation."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    """
    Simple MLP layer with ReLU activation.
    """

    def __init__(self, hidden_dim, norm="layer", learn_affine=True, track_norm=False):
        """
        Initializes the MLP Layer.

        Args:
            hidden_dim: Hidden dimension size.
            norm: Feature normalization scheme ("layer"/"batch"/None).
            learn_affine: Whether the normalizer has learnable affine parameters.
            track_norm: Whether batch statistics are used to compute normalization mean/std.
        """
        super(MLPLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.norm_type = norm
        self.learn_affine = learn_affine

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_affine),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_affine, track_running_stats=track_norm),
        }.get(norm, None)

    def forward(self, x):
        """Forward pass."""
        batch_size, num_nodes, hidden_dim = x.shape
        x_in = x

        # Linear transformation
        x = self.U(x)

        # Normalize features
        if self.norm:
            x = self.norm(x.view(batch_size * num_nodes, hidden_dim)).view(batch_size, num_nodes, hidden_dim)

        # Apply non-linearity
        x = F.relu(x)

        # Make residual connection
        x = x_in + x

        return x
