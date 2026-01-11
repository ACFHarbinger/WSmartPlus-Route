"""MLP Encoder."""
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
        self.norm = norm
        self.learn_affine = learn_affine

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_affine),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_affine, track_running_stats=track_norm)
        }.get(self.norm, None)

    def forward(self, x):
        """Forward pass."""
        batch_size, num_nodes, hidden_dim = x.shape
        x_in = x

        # Linear transformation
        x = self.U(x)

        # Normalize features
        x = self.norm(
            x.view(batch_size*num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, hidden_dim) if self.norm else x

        # Apply non-linearity
        x = F.relu(x)

        # Make residual connection
        x = x_in + x

        return x


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder with ReLU activation, independent of graph structure.
    """
    def __init__(self, n_layers, hidden_dim, norm="layer",
                 learn_affine=True, track_norm=False, *args, **kwargs):
        """
        Initializes the MLP Encoder.

        Args:
            n_layers: Number of MLP layers.
            hidden_dim: Hidden dimension size.
            norm: Normalization type.
            learn_affine: Whether to learn affine parameters.
            track_norm: Whether to track normalization stats.
        """
        super(MLPEncoder, self).__init__()
        self.layers = nn.ModuleList(
            MLPLayer(hidden_dim, norm, learn_affine, track_norm) for _ in range(n_layers)
        )

    def forward(self, x, graph=None):
        """
        Forward pass.
        
        Args:
            x: Input node features (B x V x H)
            graph: (Unused) Graph structure.
            
        Returns:
            Updated node features (B x V x H)
        """
        for layer in self.layers:
            x = layer(x)

        return x
