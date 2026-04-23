"""Pointer Network Encoder.

Attributes:
    PointerEncoder: Maps a graph represented as an input sequence to a hidden vector.

Example:
    >>> from logic.src.models.subnets.encoders.ptr import PointerEncoder
    >>> encoder = PointerEncoder(input_dim=2, hidden_dim=128)
"""

import math
from typing import Tuple

import torch
from torch import nn


class PointerEncoder(nn.Module):
    """Maps a graph represented as an input sequence to a hidden vector.

    Attributes:
        hidden_dim (int): Hidden dimension size.
        lstm (nn.LSTM): LSTM layer for sequence processing.
        init_hx (nn.Parameter): Trainable initial hidden state.
        init_cx (nn.Parameter): Trainable initial cell state.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initializes the PointerEncoder.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension.
        """
        super(PointerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.init_hx, self.init_cx = self.init_hidden(hidden_dim)

    def forward(
        self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the encoder.

        Args:
            x: Input sequence of shape (seq_len, batch, input_dim).
            hidden: Initial hidden and cell states (h_0, c_0).

        Returns:
            Tuple containing:
                - output: LSTM output sequence.
                - hidden: Final hidden and cell states (h_n, c_n).
        """
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim: int) -> Tuple[nn.Parameter, nn.Parameter]:
        """Initializes trainable initial hidden and cell states.

        Args:
            hidden_dim: Dimension of the hidden state.

        Returns:
            Tuple containing the initial hidden state (hx) and cell state (cx).
        """
        std = 1.0 / math.sqrt(hidden_dim)
        enc_init_hx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_hx.data.uniform_(-std, std)

        enc_init_cx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_cx.data.uniform_(-std, std)
        return enc_init_hx, enc_init_cx
