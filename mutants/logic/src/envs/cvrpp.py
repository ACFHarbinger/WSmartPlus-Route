"""
CVRPP Environment implementation.
"""

from __future__ import annotations

from typing import Optional

import torch
from logic.src.envs.vrpp import VRPPEnv
from tensordict import TensorDict


class CVRPPEnv(VRPPEnv):
    """
    Capacitated VRPP: VRPP with vehicle capacity constraints.
    """

    name: str = "cvrpp"

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize CVRPP state with capacity tracking."""
        td = super()._reset_instance(td)

        bs = td.batch_size[0]
        device = td.device

        # Track remaining capacity
        capacity = td.get("capacity", torch.ones(bs, device=device) * 100)
        td["capacity"] = capacity  # Ensure it's in the TensorDict for _step
        td["remaining_capacity"] = capacity.clone()
        td["collected_waste"] = torch.zeros(bs, device=device)
        td["collected"] = td["collected_waste"]  # Alias

        return td

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:  # type: ignore[override]
        # Correct TorchRL signature
        return super()._reset(tensordict, **kwargs)

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Execute action with capacity tracking."""
        # RL4CO base calls _step_instance, then updates mask.
        # So we must update our state here.
        action = td["action"]

        # Update capacity when collecting
        waste = td.get("waste", torch.zeros_like(td["remaining_capacity"].unsqueeze(-1)))
        if waste.dim() > 1:
            waste_at_node = waste.gather(1, action.unsqueeze(-1)).squeeze(-1)
        else:
            waste_at_node = waste

        # Reset capacity at depot
        at_depot = action == 0
        td["remaining_capacity"] = torch.where(
            at_depot,
            td["capacity"],
            td["remaining_capacity"] - waste_at_node,
        )
        td["collected_waste"] = torch.where(
            at_depot,
            torch.zeros_like(td["collected_waste"]),
            td["collected_waste"] + waste_at_node,
        )
        td["collected"] = td["collected_waste"]  # Alias

        return td

    def _step(self, td: TensorDict) -> TensorDict:
        # RL4CO base _step calls _step_instance and then _get_action_mask.
        # Since we use _step_instance above, we just call super().
        return super(VRPPEnv, self)._step(td)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Mask nodes that would exceed capacity, respecting must-go constraints.

        Applies capacity constraints on top of base VRPP mask, then
        re-applies must-go logic to determine depot validity.
        """
        # Get base mask (without must-go depot logic applied yet)
        base_mask = ~td["visited"].clone()

        # Mask nodes whose waste exceeds remaining capacity
        waste = td.get("waste")
        if waste is not None and waste.shape[-1] == td["visited"].shape[-1] - 1:
            # needs depot
            waste = torch.cat([torch.zeros(*td.batch_size, 1, device=td.device), waste], dim=1)

        if waste is None:
            waste = torch.zeros_like(td["visited"], dtype=torch.float32)

        remaining = td["remaining_capacity"].unsqueeze(-1)
        exceeds_capacity = waste > remaining

        mask = base_mask & ~exceeds_capacity

        # Must-go routing logic (same as parent but considering capacity)
        must_go = td.get("must_go", None)

        if must_go is not None:
            # Pending must-go bins: must_go AND not yet visited AND within capacity
            pending_must_go = must_go & mask

            # Check if any must-go bins remain (excluding depot at index 0)
            has_pending_must_go = pending_must_go[:, 1:].any(dim=1)

            # Depot is valid only if no pending must-go bins remain
            mask[:, 0] = ~has_pending_must_go
        else:
            # No must-go constraint: depot always valid
            mask[:, 0] = True

        return mask
