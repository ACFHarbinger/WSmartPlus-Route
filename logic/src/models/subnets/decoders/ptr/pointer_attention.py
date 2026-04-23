"""Pointer attention module.

This module provides the PointerAttention class, which implements the bahdanau-style
attention mechanism used by Pointer Networks.

Attributes:
    PointerAttention: Attention mechanism for pointing to specific input elements.

Example:
    >>> from logic.src.models.subnets.decoders.ptr.pointer_attention import PointerAttention
    >>> attention = PointerAttention(dim=128)
    >>> ref_proj, logits = attention(query, reference)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class PointerAttention(nn.Module):
    """Generic attention module for pointer-style decoders.

    Calculates compatibility scores between a query vector (recurrent state)
    and a reference sequence (encoder outputs) to produce selection logits.

    Attributes:
        use_tanh (bool): Whether to use tanh exploration.
        project_query (nn.Linear): Linear projection for the query vector.
        project_ref (nn.Conv1d): 1D convolution projection for the reference sequence.
        C (float): Tanh exploration scaling constant.
        tanh (nn.Tanh): Tanh activation function.
        v (nn.Parameter): Learnable attention weight vector.
    """

    def __init__(self, dim: int, use_tanh: bool = False, C: float = 10.0) -> None:
        """Initializes PointerAttention.

        Args:
            dim: Dimension of the attention vector.
            use_tanh: Whether to use tanh exploration.
            C: Tanh exploration constant.
        """
        super().__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C
        self.tanh = nn.Tanh()

        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1.0 / math.sqrt(dim)), 1.0 / math.sqrt(dim))

    def forward(self, query: torch.Tensor, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates attention logits for a given query and reference set.

        Args:
            query: Current decoder hidden state of shape (batch, dim).
            ref: Reference sequence from the encoder of shape (sourceL, batch, dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Projected references and the resulting
                attention logits.
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        logits = self.C * self.tanh(u) if self.use_tanh else u
        return e, logits
