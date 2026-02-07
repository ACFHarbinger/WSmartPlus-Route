"""
Context embeddings for decoder.

Aligns with rl4co's EnvContext pattern.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class EnvContext(nn.Module):
    """Base class for environment context embeddings."""

    def __init__(self, embed_dim: int, step_context_dim: int = 0, node_dim: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.step_context_dim = step_context_dim
        self.node_dim = node_dim

        # Projection for step context (if any)
        if step_context_dim > 0:
            self.project_context = nn.Linear(step_context_dim, embed_dim, bias=False)

    def forward(
        self,
        embeddings: torch.Tensor,
        td: Any,  # TensorDict
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Node embeddings (batch, num_loc, embed_dim)
            td: TensorDict containing state
        """
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)

        # Combine
        if state_embedding is not None:
            return cur_node_embedding + state_embedding
        return cur_node_embedding

    def _cur_node_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        """Get embedding of current node."""
        # Default implementation: gather embedding of current node
        # Assumes td has 'current_node' index
        # [batch, 1]
        cur_node = td["current_node"]
        # [batch, 1, embed_dim]
        # Gather from embeddings [batch, num_loc, embed_dim]
        # Need to handle case where current_node is not valid index (e.g. 0 at start if 0 is depot)

        # rl4co uses gather_by_index
        batch_size = embeddings.size(0)
        # Expand index: [batch, 1] -> [batch, 1, embed_dim]
        # Use simple gather for now
        # Note: current_node shape might be [batch] or [batch, 1]
        if cur_node.dim() == 1:
            cur_node = cur_node.unsqueeze(-1)

        cur_node_model = torch.gather(embeddings, 1, cur_node.unsqueeze(-1).expand(batch_size, 1, self.embed_dim))
        return cur_node_model

    def _state_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor | None:
        """Get embedding of current state."""
        return None  # To be implemented by subclasses


class VRPPContext(EnvContext):
    """Context embedding for VRPP."""

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim, step_context_dim=1)
        # Context: remaining capacity (1 dim)

    def _state_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        # VRPP state context typically includes remaining capacity
        # capacity: [batch, 1] or [batch]
        subcl = td.get("current_load", None)
        if subcl is None:
            # Try other key conventions? VRPP usually tracks load or capacity
            # For now assume 'current_load' exists or 'used_capacity'
            return torch.zeros(embeddings.size(0), 1, self.embed_dim, device=embeddings.device)

        if subcl.dim() == 1:
            subcl = subcl.unsqueeze(-1)

        out = self.project_context(subcl)
        # out is [batch, embed_dim] (if input was [batch, 1])
        # We need [batch, 1, embed_dim]
        if out.dim() == 2:
            out = out.unsqueeze(1)
        return out
