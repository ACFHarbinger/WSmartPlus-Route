"""
WCVRP Environment implementation.

Waste Collection Vehicle Routing Problem: Collect waste from bins
while respecting vehicle capacity constraints.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from logic.src.envs.base import RL4COEnvBase
from logic.src.envs.generators import WCVRPGenerator
from tensordict import TensorDict


class WCVRPEnv(RL4COEnvBase):
    """
    Waste Collection VRP Environment.

    The agent must collect waste from bins before they overflow,
    while minimizing travel cost and respecting vehicle capacity.
    """

    NAME = "wcvrp"
    name: str = "wcvrp"

    def __init__(
        self,
        generator: Optional[WCVRPGenerator] = None,
        generator_params: Optional[dict] = None,
        overflow_penalty: float = 10.0,
        collection_reward: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """
        Initialize WCVRPEnv.

        Args:
            generator: Problem instance generator.
            generator_params: Parameters for generator initialization.
            overflow_penalty: Penalty for bin overflow.
            collection_reward: Reward weight for waste collection.
            cost_weight: Weight for travel cost in reward.
            revenue_kg: Optional revenue per kg (overrides collection_reward).
            cost_km: Optional cost per kg (overrides cost_weight).
            device: Device for torch tensors ('cpu' or 'cuda').
            **kwargs: Additional keyword arguments.
        """
        generator_params = generator_params or kwargs
        if generator is None:
            generator = WCVRPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)
        self.overflow_penalty = overflow_penalty
        self.collection_reward = revenue_kg if revenue_kg is not None else collection_reward
        self.cost_weight = cost_km if cost_km is not None else cost_weight

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize WCVRP episode state."""
        device = td.device
        bs = td.batch_size
        num_loc = td["locs"].shape[-2]
        # Robustly determine number of nodes
        waste_key = "waste"
        if waste_key in td.keys() and td[waste_key].shape[-1] == num_loc:
            # Both have same size, usually means both are customer-only
            num_nodes = num_loc + 1
        else:
            num_nodes = num_loc  # Already has depot?

        td["current_node"] = torch.zeros(bs, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["visited"][..., 0] = True

        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, device=device)

        # Vehicle load tracking
        td["current_load"] = torch.zeros(*bs, device=device)
        td["collected_waste"] = torch.zeros(*bs, device=device)
        td["total_collected"] = td["collected_waste"]  # Alias

        # Initial overflows
        max_waste = td.get("max_waste", torch.tensor(1.0, device=device))
        waste = td["waste"]
        if max_waste.dim() > 1:
            max_waste = max_waste[..., 1:]
        elif max_waste.dim() == 1:
            max_waste = max_waste.unsqueeze(-1)
        td["cur_overflows"] = (waste[..., 1:] >= max_waste).float().sum(-1)

        return td

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Execute action and update state."""
        # Core mechanics
        td = super()._step_instance(td)

        action = td["action"]
        is_not_depot = action != 0
        # Combine depot and customers for full locs if not already done
        locs = td["locs"]
        if locs.shape[-2] == td["visited"].shape[-1] - 1:
            locs = torch.cat([td["depot"].unsqueeze(1), locs], dim=1)

        # Combine depot and customers for waste if not already done
        waste = td["waste"]
        if waste.shape[-1] == td["visited"].shape[-1] - 1:
            waste = torch.cat([torch.zeros(*td.batch_size, 1, device=td.device), waste], dim=1)
            td["waste"] = waste

        waste_at_node = waste.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Collect waste (clamped to max_waste if present)
        max_w = td.get("max_waste", torch.tensor(1e9, device=td.device))
        if max_w.dim() > 1 and max_w.shape[-1] == td["visited"].shape[-1] - 1:
            max_w = torch.cat([torch.tensor(1e9, device=td.device).expand(*td.batch_size, 1), max_w], dim=1)
        if max_w.dim() > 1:
            max_w_at_node = max_w.gather(1, action.unsqueeze(-1)).squeeze(-1)
        else:
            max_w_at_node = max_w

        collected = torch.min(waste_at_node, max_w_at_node) * is_not_depot.float()
        td["current_load"] = td["current_load"] + collected
        td["collected_waste"] = td["collected_waste"] + collected
        td["total_collected"] = td["collected_waste"]  # Alias

        # Empty at depot
        at_depot = action == 0
        td["current_load"] = torch.where(at_depot, torch.zeros_like(td["current_load"]), td["current_load"])

        # Clear bin (set waste to 0 after collection)
        td["waste"].scatter_(1, action.unsqueeze(-1), 0)

        # Update current node (overriding super which might used unsqueezed action)
        td["current_node"] = action.squeeze(-1) if action.dim() > 1 else action

        return td

    def _step(self, td: TensorDict) -> TensorDict:
        return super(WCVRPEnv, self)._step(td)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute action mask based on capacity, visited status, and must-go constraints.

        The must_go mask determines routing behavior:
        - If must_go is None: Standard behavior (depot always valid)
        - If must_go has True values: Must route to those bins; depot invalid until done
        - If must_go is all False: No routing needed; depot is valid (stay)

        Returns:
            Tensor: Boolean mask (batch, num_nodes) where True = valid action.
        """
        mask = ~td["visited"].clone()

        # Check capacity constraints
        waste = td["waste"]
        remaining_capacity = td["capacity"].unsqueeze(-1) - td["current_load"].unsqueeze(-1)
        exceeds_capacity = waste > remaining_capacity

        mask = mask & ~exceeds_capacity

        # Must-go routing logic
        must_go = td.get("must_go", None)

        if must_go is not None:
            # must_go: (batch, num_nodes) boolean tensor
            # True = must visit this bin, False = optional

            # Pending must-go bins: must_go AND not yet visited AND within capacity
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
        Compute WCVRP reward.

        Efficient routing (low cost, high collection) = high reward.
        """
        collection = td["total_collected"]
        cost = td["tour_length"]

        # Add return to depot
        current = td["current_node"]
        if current.dim() > 1:
            current = current.squeeze(-1)
        if current.dim() == 0:
            current = current.unsqueeze(0)

        locs = td["locs"]
        # Use robust concatenation
        if locs.shape[-2] == td.get("visited", torch.zeros(0)).shape[-1] - 1:
            locs = torch.cat([td["depot"].unsqueeze(1), locs], dim=1)

        # Use robust indexing
        current_idx = current[:, None, None].expand(-1, -1, 2)
        current_loc = locs.gather(1, current_idx).squeeze(1)
        depot_loc = td["depot"]
        return_distance = torch.norm(depot_loc - current_loc, dim=-1)

        not_at_depot = current != 0
        total_cost = cost + return_distance * not_at_depot.float()

        # Calculate overflows: unvisited nodes where waste >= max_waste
        # In current WCVRP, visited nodes have waste=0 in the final td.
        max_waste = td.get("max_waste", torch.tensor(1.0, device=td.device))

        # We only care about customers (index 1 onwards) for overflows
        waste = td["waste"][..., 1:]
        if max_waste.dim() > 1:  # (B, N)
            max_waste = max_waste[..., 1:]
        elif max_waste.dim() == 1:
            max_waste = max_waste.unsqueeze(-1)

        overflows = (waste >= max_waste).float().sum(-1)

        # Store individual components in TensorDict for logging/meta access
        td["collection"] = collection
        td["cost"] = total_cost
        td["overflows"] = overflows
        td["cur_overflows"] = overflows  # Alias

        # Store decomposed rewards for GDPO (signed for maximization)
        td["reward_collection"] = collection
        td["reward_cost"] = -total_cost
        td["reward_overflow"] = -overflows

        reward = self.collection_reward * collection - self.cost_weight * total_cost - self.overflow_penalty * overflows

        # Ensure it's 1D [B] matching batch_size
        if reward.dim() > 1:
            reward = reward.squeeze(-1)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)

        return reward.view(td.batch_size)
