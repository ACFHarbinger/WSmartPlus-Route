"""
Layers for HRL Manager.
"""

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    LSTM-based Temporal Encoder for waste history.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Input size is 1 because we reshape dynamic features to (Batch*N, History, 1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)

    def forward(self, dynamic: torch.Tensor) -> torch.Tensor:
        """
        Encode dynamic waste history.

        Args:
            dynamic: (Batch, N, History)

        Returns:
            temporal_embed: (Batch, N, hidden_dim)
        """
        B, N, H_len = dynamic.size()
        # Reshape dynamic for LSTM: (Batch*N, History, 1)
        dyn_flat = dynamic.view(B * N, H_len, 1)
        _, (h_n, _) = self.lstm(dyn_flat)
        return h_n.squeeze(0).view(B, N, -1)


class MustGoHead(nn.Module):
    """
    Head for selecting must-go nodes.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GateHead(nn.Module):
    """
    Head for routing decision (gate).
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticHead(nn.Module):
    """
    Critic head for value estimation.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
