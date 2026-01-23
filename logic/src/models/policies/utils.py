"""
Shared utilities for Policies.
"""

from typing import Optional

import torch
from tensordict import TensorDict


class TensorDictStateWrapper:
    """
    Wraps a TensorDict to expose methods expected by legacy model components (Decoders).

    Acts as a bridge between the new TensorDict state management and the
    legacy object-oriented state access patterns (e.g. state.get_mask()).
    """

    def __init__(self, td: TensorDict, problem_name: str = "vrpp"):
        """Initialize TensorDictStateWrapper."""
        self.td = td
        self.problem_name = problem_name

        # Expose common properties directly
        self.dist_matrix = td.get("dist", None)

        # Handle 'demands_with_depot' for WCVRP partial updates
        if "demand" in td.keys():
            self.demands_with_depot = td["demand"]

    def get_mask(self) -> Optional[torch.Tensor]:
        """Get action mask from TensorDict."""
        # RL4CO envs provide "action_mask" where True means VALID.
        # AttentionDecoder expects mask where True means INVALID (masked out).
        # So we invert it.
        if "action_mask" in self.td.keys():
            mask = ~self.td["action_mask"]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            return mask
        return None

    def get_edges_mask(self) -> Optional[torch.Tensor]:
        """Get edge mask from TensorDict."""
        return self.td.get("graph_mask", None)

    def get_current_node(self) -> torch.Tensor:
        """Get current node (last visited)."""
        return self.td["current_node"].long()

    def get_current_profit(self) -> torch.Tensor:
        """For VRPP: get cumulative collected prize."""
        # This is used for context embedding.
        val = self.td.get("collected_prize", torch.zeros(self.td.batch_size, device=self.td.device))
        if val.dim() == 1:
            val = val.unsqueeze(-1)
        return val

    def get_current_efficiency(self) -> torch.Tensor:
        """For WCVRP: get current efficiency."""
        # Legacy placeholder
        val = torch.zeros(self.td.batch_size, device=self.td.device)
        return val.unsqueeze(-1)

    def get_remaining_overflows(self) -> torch.Tensor:
        """For WCVRP: get remaining overflows."""
        # Legacy placeholder
        val = torch.zeros(self.td.batch_size, device=self.td.device)
        return val.unsqueeze(-1)


class DummyProblem:
    """Minimal problem wrapper for legacy component initialization."""

    def __init__(self, name: str):
        """Initialize DummyProblem."""
        self.NAME = name
