"""
WCVRP Environment implementation.

Waste Collection Vehicle Routing Problem: Collect waste from bins
while respecting vehicle capacity constraints.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.envs.generators import WCVRPGenerator


class WCVRPEnv(RL4COEnvBase):
    """
    Waste Collection VRP Environment.

    The agent must collect waste from bins before they overflow,
    while minimizing travel cost and respecting vehicle capacity.
    """

    name: str = "wcvrp"

    def __init__(
        self,
        generator: Optional[WCVRPGenerator] = None,
        generator_params: Optional[dict] = None,
        overflow_penalty: float = 10.0,
        collection_reward: float = 1.0,
        cost_weight: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        generator_params = generator_params or {"num_loc": 50}
        if generator is None:
            generator = WCVRPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)
        self.overflow_penalty = overflow_penalty
        self.collection_reward = collection_reward
        self.cost_weight = cost_weight

    def _reset(self, td: TensorDict, batch_size: Optional[Union[int, list[int], tuple[int, ...]]] = None) -> TensorDict:
        """Initialize WCVRP episode state."""
        device = td.device
        if batch_size is None:
            bs = td.batch_size
        elif isinstance(batch_size, int):
            bs = (batch_size,)
        else:
            bs = tuple(batch_size)

        num_nodes = td["locs"].shape[-2]

        td["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["visited"][..., 0] = True

        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, device=device)

        # Vehicle load tracking
        td["current_load"] = torch.zeros(*bs, device=device)
        td["total_collected"] = torch.zeros(*bs, device=device)

        td["i"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)

        return td

    def _step(self, td: TensorDict) -> TensorDict:
        """Execute action and update state."""
        action = td["action"]
        current = td["current_node"].squeeze(-1)
        locs = td["locs"]

        # Compute distance
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        td["tour_length"] = td["tour_length"] + distance

        # Collection logic
        is_not_depot = action != 0
        demand_at_node = td["demand"].gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Collect waste
        collected = demand_at_node * is_not_depot.float()
        td["current_load"] = td["current_load"] + collected
        td["total_collected"] = td["total_collected"] + collected

        # Empty at depot
        at_depot = action == 0
        td["current_load"] = torch.where(at_depot, torch.zeros_like(td["current_load"]), td["current_load"])

        # Clear bin (set demand to 0 after collection)
        new_demand = td["demand"].clone()
        new_demand.scatter_(1, action.unsqueeze(-1), 0)
        td["demand"] = new_demand

        # Update state
        td["visited"] = td["visited"].scatter(1, action.unsqueeze(-1), True)
        td["current_node"] = action.unsqueeze(-1)
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)
        td["i"] = td["i"] + 1

        return td

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Mask based on capacity and visited status."""
        mask = ~td["visited"].clone()

        # Check capacity constraints
        demand = td["demand"]
        remaining_capacity = td["capacity"].unsqueeze(-1) - td["current_load"].unsqueeze(-1)
        exceeds_capacity = demand > remaining_capacity

        mask = mask & ~exceeds_capacity
        mask[:, 0] = True  # Can always go to depot

        return mask

    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute WCVRP reward.

        Efficient routing (low cost, high collection) = high reward.
        """
        collection = td["total_collected"]
        cost = td["tour_length"]

        # Add return to depot
        current = td["current_node"].squeeze(-1)
        locs = td["locs"]
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        depot_loc = td["depot"]
        return_distance = torch.norm(depot_loc - current_loc, dim=-1)

        not_at_depot = current != 0
        total_cost = cost + return_distance * not_at_depot.float()

        # Store individual components in TensorDict for logging/meta access
        td["collection"] = collection
        td["cost"] = total_cost

        reward = self.collection_reward * collection - self.cost_weight * total_cost
        return reward


class CWCVRPEnv(WCVRPEnv):
    """Capacitated WCVRP with strict capacity constraints."""

    name: str = "cwcvrp"


class SDWCVRPEnv(WCVRPEnv):
    """Stochastic Demand WCVRP with uncertain fill rates."""

    name: str = "sdwcvrp"

    def _step(self, td: TensorDict) -> TensorDict:
        """Step with stochastic demand simulation."""
        td = super()._step(td)

        # Simulate fill rate increase for unvisited bins
        if "fill_rates" in td.keys():
            fill_increase = td["fill_rates"] * (~td["visited"]).float()
            td["demand"] = (td["demand"] + fill_increase).clamp(max=1.0)

        return td
