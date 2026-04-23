"""Meta Recurrent Neural Network for Dynamic Reward Adjustment.

This module implements the `WeightAdjustmentRNN`, a meta-learning model
designed to identify patterns in how reward weight modifications influence
long-term agent performance, enabling automated multi-objective tuning.

Attributes:
    WeightAdjustmentRNN: Recurrent meta-model for objective weighting.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class WeightAdjustmentRNN(nn.Module):
    """LSTM-based Dynamic Weight Adjustment Model.

    Analyzes a sequence of historical performance metrics and corresponding
    reward weights to predict optimal adjustments for the next optimization
    period.

    Attributes:
        hidden_size (int): dimensional size of the recurrent state.
        num_layers (int): depth of the LSTM network.
        lstm (nn.LSTM): temporal processing core.
        fc (nn.Sequential): projection layers for weight adjustment prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
    ) -> None:
        """Initializes the weight adjustment RNN.

        Args:
            input_size: size of input features (current weights + metrics).
            hidden_size: dimensional width of the LSTM hidden state.
            output_size: number of objective weights to adjust.
            num_layers: number of stacked recurrent layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM for capturing temporal patterns in weight adjustments
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Fully connected layers for weight adjustment prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initializes parameters using Xavier normal for training stability."""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predicts weight adjustments from historical performance sequence.

        Args:
            x: Input feature sequence [batch_size, seq_len, input_size].
            hidden: Initial hidden/cell states for the LSTM.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - weight_adjustments: Suggested delta for each weight [batch_size, output_size].
                - hidden: Updated LSTM hidden and cell states.
        """
        lstm_out, hidden = self.lstm(x, hidden)

        # Extract features from the final sequence step
        last_output = lstm_out[:, -1, :]

        # Predict additive/multiplicative weight adjustments
        weight_adjustments = self.fc(last_output)
        return weight_adjustments, hidden
