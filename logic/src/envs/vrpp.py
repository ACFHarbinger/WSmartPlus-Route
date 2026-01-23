"""
VRPP Environment implementation.

Vehicle Routing Problem with Profits: Select profitable subset
of nodes to visit while minimizing travel cost.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.envs.generators import VRPPGenerator


class VRPPEnv(RL4COEnvBase):
    """
    Vehicle Routing Problem with Profits Environment.

    The agent must select which nodes to visit to maximize
    total prize collected minus travel cost.
    """

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
        self.prize_weight = revenue_kg if revenue_kg is not None else prize_weight
        self.cost_weight = cost_km if cost_km is not None else cost_weight

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize VRPP episode state."""
        device = td.device
        bs = td.batch_size
        num_nodes = td["locs"].shape[-2]

        # Initialize state fields
        td["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["visited"][..., 0] = True  # Depot is "visited" initially

        # Tour tracking
        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, device=device)
        td["collected_prize"] = torch.zeros(*bs, device=device)

        return td

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Execute action and update state."""
        action = td["action"]
        current = td["current_node"].squeeze(-1)
        locs = td["locs"]

        # Compute distance traveled
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        # Update tour length
        td["tour_length"] = td["tour_length"] + distance

        # Collect prize (only for unvisited, non-depot nodes)
        is_new_visit = ~td["visited"].gather(1, action.unsqueeze(-1)).squeeze(-1)
        is_not_depot = action != 0
        prize_collected = td["prize"].gather(1, action.unsqueeze(-1)).squeeze(-1)
        td["collected_prize"] = td["collected_prize"] + prize_collected * is_new_visit * is_not_depot

        # Update visited
        td["visited"] = td["visited"].scatter(1, action.unsqueeze(-1), True)

        # Update current node
        td["current_node"] = action.unsqueeze(-1)

        # Append to tour
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute action mask for VRPP.

        - Can visit any unvisited node
        - Can return to depot at any time
        - Cannot visit already visited nodes (except depot)
        """
        mask = ~td["visited"].clone()
        mask[:, 0] = True  # Can always return to depot
        return mask

    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute VRPP reward: prize - cost.

        Reward = (prize_weight * total_prize) - (cost_weight * tour_length)
        """
        prize = td["collected_prize"] if "collected_prize" in list(td.keys()) else torch.zeros_like(td["tour_length"])
        cost = td["tour_length"]

        # Also add return to depot distance if not already there
        current = td["current_node"].squeeze(-1)
        locs = td["locs"]
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        depot_loc = td["depot"]
        return_distance = torch.norm(depot_loc - current_loc, dim=-1)

        # Only add return distance if not already at depot
        not_at_depot = current != 0
        total_cost = cost + return_distance * not_at_depot.float()

        # Store decomposed rewards for GDPO
        td["reward_prize"] = prize
        td["reward_cost"] = -total_cost  # Convention: maximized, so negative cost

        reward = self.prize_weight * prize - self.cost_weight * total_cost
        return reward

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Episode is done when agent returns to depot."""
        current = td["current_node"].squeeze(-1)
        step = td["i"].squeeze(-1)
        return (current == 0) & (step > 0)


class CVRPPEnv(VRPPEnv):
    """
    Capacitated VRPP: VRPP with vehicle capacity constraints.
    """

    name: str = "cvrpp"

    def _reset(self, td: TensorDict, batch_size: Optional[int] = None) -> TensorDict:
        """Initialize CVRPP state with capacity tracking."""
        td = super()._reset(td, batch_size)

        bs = td.batch_size[0] if batch_size is None else batch_size
        device = td.device

        # Track remaining capacity
        capacity = td.get("capacity", torch.ones(bs, device=device) * 100)
        td["remaining_capacity"] = capacity.clone()
        td["collected"] = torch.zeros(bs, device=device)

        return td

    def _step(self, td: TensorDict) -> TensorDict:
        """Execute action with capacity tracking."""
        td = super()._step(td)
        action = td["action"]

        # Update capacity when collecting
        demand = td["demand"].gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Reset capacity at depot
        at_depot = action == 0
        td["remaining_capacity"] = torch.where(
            at_depot,
            td["capacity"],
            td["remaining_capacity"] - demand,
        )
        td["collected"] = torch.where(
            at_depot,
            torch.zeros_like(td["collected"]),
            td["collected"] + demand,
        )

        return td

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Mask nodes that would exceed capacity."""
        mask = super()._get_action_mask(td)

        # Mask nodes whose demand exceeds remaining capacity
        demand = td["demand"]
        remaining = td["remaining_capacity"].unsqueeze(-1)
        exceeds_capacity = demand > remaining

        mask = mask & ~exceeds_capacity
        mask[:, 0] = True  # Can always return to depot

        return mask
