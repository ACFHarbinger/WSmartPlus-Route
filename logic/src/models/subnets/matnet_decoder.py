from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from logic.src.models.subnets.glimpse_decoder import GlimpseDecoder


class MatNetDecoder(GlimpseDecoder):
    """
    Decoder for MatNet.
    Extends GlimpseDecoder to handle row and column embeddings from MatNetEncoder.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        n_heads: int = 8,
        tanh_clipping: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
            **kwargs,
        )
        # MatNet context might need both row and col info
        # Standard context is [batch, embed_dim]
        # For MatNet, we might project both
        self.project_row_context = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_col_context = nn.Linear(embed_dim, embed_dim, bias=False)

    def _precompute(self, row_embeddings: torch.Tensor, col_embeddings: torch.Tensor) -> Any:
        """
        Precompute fixed elements.
        Args:
            row_embeddings: [batch, row_size, embed_dim]
            col_embeddings: [batch, col_size, embed_dim]
        """
        # For MatNet, we use row_embeddings as the primary node embeddings for the decoder
        # (assuming we are picking rows/nodes)
        fixed = super()._precompute(row_embeddings)

        # Add column information to fixed context
        col_avg = col_embeddings.mean(1)
        fixed.context_node_projected = fixed.context_node_projected + self.project_col_context(col_avg)[:, None, :]

        return fixed

    def forward(
        self,
        input_data: Union[torch.Tensor, dict[str, torch.Tensor]],
        row_embeddings: torch.Tensor,
        col_embeddings: torch.Tensor,
        cost_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Matrix-aware forward pass.
        """
        # We need to override _inner or provide a way to pass both row and col embeddings
        # For now, we'll implement a simplified forward that sets up the state and fixed context

        state = self.problem.make_state(input_data, cost_weights=cost_weights)
        fixed = self._precompute(row_embeddings, col_embeddings)

        # Use standard GlimpseDecoder loop for now
        # Note: This might need more customization for FFSP (multi-stage)
        outputs = []
        sequences = []

        i = 0
        while not state.all_finished():
            log_p, mask = self._get_log_p(fixed, state)
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
            state = state.update(selected)
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
            i += 1

        return torch.stack(outputs, 1), torch.stack(sequences, 1)
