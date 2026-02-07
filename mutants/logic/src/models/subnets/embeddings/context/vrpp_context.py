"""VRPP specific context embedding."""

from __future__ import annotations

from typing import Any

import torch

from .context_base import EnvContext


class VRPPContext(EnvContext):
    """Context embedding for VRPP."""

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim, step_context_dim=1)

    def _state_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        # Remaining length or capacity
        # We use 'max_length' or 'remaining_capacity'
        if "remaining_length" in td.keys():
            state = td["remaining_length"]
        elif "remaining_capacity" in td.keys():
            state = td["remaining_capacity"]
        else:
            state = torch.ones(embeddings.size(0), device=embeddings.device)

        if state.dim() == 1:
            state = state[:, None]

        return state

    def forward(self, state: Any) -> torch.Tensor:
        """Compute the forward pass for the context embedding."""
        # This base implementation might be needed or overridden by CVRPP
        # But if it was in vrpp.py, let's keep it functional.
        # Actually EnvContext forward is abstract.
        # Let's check the original vrpp.py forward.
        # Wait, the original vrpp.py VRPPContext didn't have a forward method!
        # It had _state_embedding.
        # That means it's likely used in a way that calls _state_embedding elsewhere?
        # No, EnvContext forward is abstract.
        # Let's re-read vrpp.py VRPPContext.
        pass
