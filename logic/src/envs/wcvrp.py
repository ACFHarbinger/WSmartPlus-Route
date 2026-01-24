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
            cost_km: Optional cost per km (overrides cost_weight).
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
        num_nodes = td["locs"].shape[-2]

        td["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["visited"][..., 0] = True

        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, device=device)

        # Vehicle load tracking
        td["current_load"] = torch.zeros(*bs, device=device)
        td["total_collected"] = torch.zeros(*bs, device=device)
        td["collected_prize"] = td["total_collected"]  # Alias for compatibility

        # Initial overflows
        max_waste = td.get("max_waste", torch.tensor(1.0, device=device))
        demand = td["demand"]
        if max_waste.dim() > 1:
            max_waste = max_waste[..., 1:]
        elif max_waste.dim() == 1:
            max_waste = max_waste.unsqueeze(-1)
        td["cur_overflows"] = (demand[..., 1:] >= max_waste).float().sum(-1)

        return td

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Execute action and update state."""
        # Core mechanics
        td = super()._step_instance(td)

        action = td["action"]
        is_not_depot = action != 0
        demand_at_node = td["demand"].gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Collect waste (clamped to max_waste if present)
        max_w = td.get("max_waste", torch.tensor(1e9, device=td.device))
        if max_w.dim() > 1:
            max_w_at_node = max_w.gather(1, action.unsqueeze(-1)).squeeze(-1)
        else:
            max_w_at_node = max_w

        collected = torch.min(demand_at_node, max_w_at_node) * is_not_depot.float()
        td["current_load"] = td["current_load"] + collected
        td["total_collected"] = td["total_collected"] + collected
        td["collected_prize"] = td["total_collected"]  # Alias

        # Empty at depot
        at_depot = action == 0
        td["current_load"] = torch.where(at_depot, torch.zeros_like(td["current_load"]), td["current_load"])

        # Clear bin (set demand to 0 after collection)
        # We MUST do this on td['demand'] which will be in td_next
        td["demand"].scatter_(1, action.unsqueeze(-1), 0)

        # Note: visited and current_node are updated in super()._step_instance

        return td

    def _step(self, td: TensorDict) -> TensorDict:
        return super(WCVRPEnv, self)._step(td)

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

        # Calculate overflows: unvisited nodes where demand >= max_waste
        # In current WCVRP, visited nodes have demand=0 in the final td.
        max_waste = td.get("max_waste", torch.tensor(1.0, device=td.device))

        # We only care about customers (index 1 onwards) for overflows
        demand = td["demand"][..., 1:]
        if max_waste.dim() > 1:  # (B, N)
            max_waste = max_waste[..., 1:]
        elif max_waste.dim() == 1:
            max_waste = max_waste.unsqueeze(-1)

        overflows = (demand >= max_waste).float().sum(-1)

        # Store individual components in TensorDict for logging/meta access
        td["collection"] = collection
        td["collected_prize"] = collection  # Alias
        td["cost"] = total_cost
        td["overflows"] = overflows
        td["cur_overflows"] = overflows  # Alias

        # Store decomposed rewards for GDPO (signed for maximization)
        td["reward_collection"] = collection
        td["reward_cost"] = -total_cost
        td["reward_overflow"] = -overflows

        reward = self.collection_reward * collection - self.cost_weight * total_cost - self.overflow_penalty * overflows
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
