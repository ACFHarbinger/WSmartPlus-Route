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
        """
        Initialize EnvContext.

        Args:
            embed_dim: Embedding dimension.
            step_context_dim: Dimension of step context.
            node_dim: Dimension of node features.
        """
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
        cur_node = td["current_node"]
        batch_size = embeddings.size(0)

        if cur_node.dim() == 1:
            cur_node = cur_node.unsqueeze(-1)

        # [batch, 1, embed_dim]
        cur_node_model = torch.gather(embeddings, 1, cur_node.unsqueeze(-1).expand(batch_size, 1, self.embed_dim))
        return cur_node_model

    def _state_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor | None:
        """Get embedding of current state."""
        return None  # To be implemented by subclasses


class VRPPContext(EnvContext):
    """
    Context embedding for VRPP.
    Projects remaining capacity / current load.
    """

    def __init__(self, embed_dim: int):
        """
        Initialize VRPPContext.

        Args:
            embed_dim: Embedding dimension.
        """
        super().__init__(embed_dim, step_context_dim=1)

    def _state_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        # Check standard keys for capacity/load using keys() safely
        keys = td.keys()
        if "remaining_capacity" in keys:
            # Normalize? Usually assumes normalized env input or raw projection
            # [batch, 1]
            feat = td["remaining_capacity"]
        elif "current_load" in keys:
            feat = td["current_load"]
        else:
            # Fallback
            return torch.zeros(embeddings.size(0), 1, self.embed_dim, device=embeddings.device)

        if feat.dim() == 1:
            feat = feat.unsqueeze(-1)

        out = self.project_context(feat)
        if out.dim() == 2:
            out = out.unsqueeze(1)
        return out


class CVRPContext(VRPPContext):
    """
    Context embedding for CVRP.
    Same logic as VRPP (capacity context).
    """

    pass


class WCVRPContext(VRPPContext):
    """
    Context embedding for WCVRP.
    Same logic as VRPP (capacity context).
    """

    pass


class SWCVRPContext(WCVRPContext):
    """
    Context embedding for SWCVRP (Stochastic WCVRP).
    Currently shares WCVRP context (capacity-based).
    Can be extended to include belief state / variance embeddings.
    """

    pass
