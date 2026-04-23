"""Temporal Encoder for Historical Waste Pattern Recognition.

This module provides the `TemporalEncoder`, which processes time-series waste
level data using Recurrent Neural Networks to identify fill-rate trends.

Attributes:
    TemporalEncoder: RNN-based history decoder.
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """Auto-Regressive Temporal Encoder for waste history.

    Maps a window of historical fill levels for each node into a latent temporal
    embedding that captures accumulation trends.

    Attributes:
        rnn_type (str): model architecture ('lstm' or 'gru').
        rnn (nn.Module): recurrent layer processing sequence data.
    """

    def __init__(self, hidden_dim: int = 64, rnn_type: str = "lstm") -> None:
        """Initializes the temporal encoder.

        Args:
            hidden_dim: size of the output temporal latent vector.
            rnn_type: choice of RNN cell ('lstm' or 'gru').

        Raises:
            ValueError: If an unsupported RNN type is provided.
        """
        super().__init__()
        self.rnn_type = rnn_type.lower()
        # Input size is 1 because we analyze sequence of scalar fill levels
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

    def forward(self, dynamic: torch.Tensor) -> torch.Tensor:
        """Encodes dynamic waste history into latent embeddings.

        Reshapes [B, N, H] features into a sequence of [B*N, H, 1] for efficient
        parallel RNN processing across all nodes in the batch.

        Args:
            dynamic: historical fill levels [B, N, History].

        Returns:
            torch.Tensor: temporal latent embeddings [B, N, hidden_dim].
        """
        B, N, H_len = dynamic.size()
        # Reshape for parallel RNN processing: (Batch*Nodes, History, 1)
        dyn_flat = dynamic.view(B * N, H_len, 1)

        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(dyn_flat)
        else:
            _, h_n = self.rnn(dyn_flat)

        # Restore batch and node dimensions: (Batch, Nodes, Latent)
        return h_n.squeeze(0).view(B, N, -1)
