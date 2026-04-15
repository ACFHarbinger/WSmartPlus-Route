"""
CVRPP Environment implementation.
"""

from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict

from logic.src.envs.vrpp import VRPPEnv


class CVRPPEnv(VRPPEnv):
    """
    Capacitated VRPP: VRPP with vehicle capacity constraints.
    """

    name: str = "cvrpp"

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
        """Initialize CVRPP state with capacity tracking."""
        tensordict = super()._reset_instance(tensordict)

        bs = tensordict.batch_size[0]
        device = tensordict.device

        # Track remaining capacity
        capacity = tensordict.get("capacity", torch.ones(bs, device=device) * 100)
        tensordict["capacity"] = capacity  # Ensure it's in the TensorDict for _step
        tensordict["remaining_capacity"] = capacity.clone()
        tensordict["collected_waste"] = torch.zeros(bs, device=device)
        tensordict["collected"] = tensordict["collected_waste"]  # Alias

        return tensordict

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:  # type: ignore[override]
        # Correct TorchRL signature
        return super()._reset(tensordict, **kwargs)

    def _step_instance(self, tensordict: TensorDict) -> TensorDict:
        """reset.

        Args:
            tensordict (Optional[TensorDict]): Description of tensordict.
            kwargs (Any): Description of kwargs.

        Returns:
            Any: Description of return value.
        """
        """Execute action with capacity tracking."""
        action = tensordict["action"]

        # Update capacity when collecting
        waste = tensordict["waste"]
        waste_at_node = waste.gather(1, action.unsqueeze(-1)).squeeze(-1) if waste.dim() > 1 else waste

        # Reset capacity at depot
        at_depot = action == 0
        tensordict["remaining_capacity"] = torch.where(
            at_depot,
            tensordict["capacity"],
            tensordict["remaining_capacity"] - waste_at_node,
        )
        tensordict["collected_waste"] = torch.where(
            at_depot,
            torch.zeros_like(tensordict["collected_waste"]),
            tensordict["collected_waste"] + waste_at_node,
        )
        tensordict["collected"] = tensordict["collected_waste"]  # Alias

        return tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """step.

        Args:
            tensordict (TensorDict): Description of tensordict.

        Returns:
            Any: Description of return value.
        """
        # RL4CO base _step calls _step_instance and then _get_action_mask.
        # Since we use _step_instance above, we just call super().
        return super(VRPPEnv, self)._step(tensordict)

    def _get_action_mask(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Mask nodes that would exceed capacity, respecting mandatory constraints.

        Applies capacity constraints on top of base VRPP mask, then
        re-applies mandatory logic to determine depot validity.
        """
        # Get base mask (without mandatory depot logic applied yet)
        base_mask = ~tensordict["visited"].clone()

        # Mask nodes whose waste exceeds remaining capacity
        waste = tensordict["waste"]
        remaining = tensordict["remaining_capacity"].unsqueeze(-1)
        exceeds_capacity = waste > remaining

        mask = base_mask & ~exceeds_capacity

        # Mandatory routing logic (same as parent but considering capacity)
        mandatory = tensordict.get("mandatory", None)
        if mandatory is not None:
            # Handle dimension mismatch (if mandatory excludes depot)
            if mandatory.size(-1) == mask.size(-1) - 1:
                mandatory = torch.cat([torch.zeros_like(mandatory[:, :1], dtype=torch.bool), mandatory], dim=1)

            # Pending mandatory bins: mandatory AND not yet visited AND within capacity
            pending_mandatory = mandatory & mask

            # Check if any mandatory bins remain (excluding depot at index 0)
            has_pending_mandatory = pending_mandatory[:, 1:].any(dim=1)

            # Depot is valid only if no pending mandatory bins remain
            mask[:, 0] = ~has_pending_mandatory
        else:
            # No mandatory constraint: depot always valid
            mask[:, 0] = True

        return mask
