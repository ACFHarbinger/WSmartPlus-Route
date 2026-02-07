from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from logic.src.models.subnets.matnet_decoder import MatNetDecoder
from logic.src.models.subnets.matnet_encoder import MatNetEncoder


class MatNetPolicy(nn.Module):
    """
    MatNet Policy for matrix-based Combinatorial Optimization.
    Unifies MatNetEncoder and MatNetDecoder.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        num_layers: int = 5,
        n_heads: int = 8,
        tanh_clipping: float = 10.0,
        normalization: str = "instance",
        **kwargs,
    ):
        super(MatNetPolicy, self).__init__()
        self.problem = problem
        self.embed_dim = embed_dim

        # Initial projection of matrix rows and columns
        # Assuming input is [batch, row_size, col_size]
        self.row_init_proj = nn.Linear(1, embed_dim)
        self.col_init_proj = nn.Linear(1, embed_dim)

        self.encoder = MatNetEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            n_heads=n_heads,
            feed_forward_hidden=hidden_dim,
            normalization=normalization,
        )

        self.decoder = MatNetDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
            **kwargs,
        )

    def set_decode_type(self, decode_type: str, temp: Optional[float] = None):
        self.decoder.set_decode_type(decode_type, temp)

    def forward(
        self,
        input_data: Dict[str, Any],
        cost_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Policy forward pass.
        Args:
            input_data: Dict containing 'dist' or 'cost_matrix' of shape [batch, row_size, col_size]
        """
        # Get matrix (e.g. distance or cost matrix)
        # In MatNet, we typically solve problems like ATSP or FFSP where cost is a matrix.
        matrix = input_data.get("dist") or input_data.get("cost_matrix")
        if matrix is None:
            raise ValueError("MatNetPolicy requires a cost matrix in 'dist' or 'cost_matrix' key.")

        if matrix.dim() == 2:
            matrix = matrix.unsqueeze(0)

        batch_size, row_size, col_size = matrix.size()

        # Initial row and column features (e.g., mean of rows/cols)
        # MatNet actually projects the whole row/col if they have features.
        # If it's just the matrix, we use means or individual elements.
        # Baseline: row/col means
        row_init = matrix.mean(dim=2, keepdim=True)  # [batch, row_size, 1]
        col_init = matrix.mean(dim=1, keepdim=True).transpose(1, 2)  # [batch, col_size, 1]

        row_emb = self.row_init_proj(row_init)
        col_emb = self.col_init_proj(col_init)

        # Encoding
        row_emb, col_emb = self.encoder(row_emb, col_emb)

        # Decoding
        log_p, actions = self.decoder(input_data, row_emb, col_emb, cost_weights=cost_weights, **kwargs)

        # To match standard policy interface, we might need to return (cost, ll, ...)
        # but the decoder currently returns (log_p, actions).
        # Usually, the AttentionModel wrapper handles the cost calculation.
        # We can return these directly for now or wrap it similarly.

        return log_p, actions
