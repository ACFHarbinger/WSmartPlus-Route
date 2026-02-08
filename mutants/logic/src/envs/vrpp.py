"""
VRPP Environment implementation.

Vehicle Routing Problem with Profits: Select profitable subset
of nodes to visit while minimizing travel cost.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from logic.src.envs.base import RL4COEnvBase
from logic.src.envs.generators import VRPPGenerator
from tensordict import TensorDict


class VRPPEnv(RL4COEnvBase):
    """
    Vehicle Routing Problem with Profits Environment.

    The agent must select which nodes to visit to maximize
    total prize collected minus travel cost.
    """

    NAME = "vrpp"
    name: str = "vrpp"

    def __init__(
        self,
        generator: Optional[VRPPGenerator] = None,
        generator_params: Optional[dict] = None,
        prize_weight: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """
        Initialize VRPPEnv.

        Args:
            generator: Problem instance generator.
            generator_params: Parameters for generator initialization.
            prize_weight: Weight for prize collection in reward.
            cost_weight: Weight for travel cost in reward.
            revenue_kg: Optional revenue per kg (overrides prize_weight).
            cost_km: Optional cost per km (overrides cost_weight).
            device: Device for torch tensors ('cpu' or 'cuda').
            **kwargs: Additional keyword arguments.
        """
        generator_params = generator_params or kwargs
        if generator is None:
            generator = VRPPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)
        self.waste_weight = revenue_kg if revenue_kg is not None else prize_weight
        self.cost_weight = cost_km if cost_km is not None else cost_weight

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize VRPP episode state."""
        device = td.device
        bs = td.batch_size

        # Robustly determine number of nodes
        num_loc = td["locs"].shape[-2]
        # waste/demand might already include depot (size N+1) or not (size N)
        waste_key = "waste"
        if waste_key in td.keys() and td[waste_key].shape[-1] == num_loc:
            # Both have same size, usually means both are customer-only or both already have depot
            # We assume they are customer-only if we are using separate depot
            num_nodes = num_loc + 1
        else:
            num_nodes = num_loc  # Already has depot? Or no waste info.

        # Initialize state fields
        td["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["visited"][..., 0] = True  # Depot index 0

        # Tour tracking
        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, device=device)
        td["collected_waste"] = torch.zeros(*bs, device=device)
        return td

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Execute action and update state."""
        action = td["action"]
        current = td["current_node"].squeeze(-1)
        device = td.device
        bs = td.batch_size

        # Combine depot and customers for full locs if not already done
        locs = td["locs"]
        if locs.shape[-2] == td["visited"].shape[-1] - 1:
            locs = torch.cat([td["depot"].unsqueeze(1), locs], dim=1)

        # Compute distance traveled
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        # Update tour length
        td["tour_length"] = td["tour_length"] + distance

        # Collect prize (only for unvisited, non-depot nodes)
        is_new_visit = ~td["visited"].gather(1, action.unsqueeze(-1)).squeeze(-1)
        is_not_depot = action != 0

        # waste handling
        waste = td.get("waste", td.get("prize", td.get("demand")))
        if waste is not None and waste.shape[-1] == td["visited"].shape[-1] - 1:
            # needs depot
            waste = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)

        if waste is not None:
            waste_collected = waste.gather(1, action.unsqueeze(-1)).squeeze(-1)
            td["collected_waste"] = (
                td["collected_waste"] + waste_collected * is_new_visit.float() * is_not_depot.float()
            )

        # Update visited
        td["visited"] = td["visited"].scatter(1, action.unsqueeze(-1), True)

        # Update current node
        td["current_node"] = action.unsqueeze(-1)

        # Append to tour
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute action mask for VRPP with must-go constraints.

        Standard behavior:
        - Can visit any unvisited node
        - Can return to depot at any time
        - Cannot visit already visited nodes (except depot)

        Must-go behavior:
        - If must_go is None: Standard behavior (depot always valid)
        - If must_go has True values: Must route to those bins; depot invalid until done
        - If must_go is all False: No routing needed; depot is valid (stay)

        Returns:
            Tensor: Boolean mask (batch, num_nodes) where True = valid action.
        """
        mask = ~td["visited"].clone()

        # Must-go routing logic
        must_go = td.get("must_go", None)

        if must_go is not None:
            # must_go: (batch, num_nodes) boolean tensor
            # True = must visit this bin, False = optional

            # Pending must-go bins: must_go AND not yet visited
            pending_must_go = must_go & mask

            # Check if any must-go bins remain (excluding depot at index 0)
            has_pending_must_go = pending_must_go[:, 1:].any(dim=1)

            # Depot is valid only if no pending must-go bins remain
            # If has_pending_must_go is True -> depot invalid (False)
            # If has_pending_must_go is False -> depot valid (True)
            mask[:, 0] = ~has_pending_must_go
        else:
            # No must-go constraint: depot always valid
            mask[:, 0] = True

        return mask

    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute VRPP reward: waste - cost.

        Reward = (prize_weight * total_waste) - (cost_weight * tour_length)
        """
        waste = td.get("collected_waste", torch.zeros_like(td["tour_length"]))
        cost = td["tour_length"]

        # Also add return to depot distance if not already there
        current = td["current_node"].squeeze(-1)
        # Combine depot and customers for full locs if not already done
        locs = td["locs"]
        if locs.shape[-2] == td.get("visited", torch.zeros(0)).shape[-1] - 1:
            locs = torch.cat([td["depot"].unsqueeze(1), locs], dim=1)

        current_idx = current[:, None, None].expand(-1, -1, 2)
        current_loc = locs.gather(1, current_idx).squeeze(1)
        depot_loc = td["depot"]
        return_distance = torch.norm(depot_loc - current_loc, dim=-1)

        # Only add return distance if not already at depot
        not_at_depot = current != 0
        total_cost = cost + return_distance * not_at_depot.float()

        # Store decomposed rewards for GDPO
        td["reward_waste"] = waste
        td["reward_cost"] = -total_cost  # Convention: maximized, so negative cost

        reward = self.waste_weight * waste - self.cost_weight * total_cost
        return reward

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Episode is done when agent returns to depot."""
        current = td["current_node"].squeeze(-1)
        step = td["i"].squeeze(-1)
        return (current == 0) & (step > 0)
