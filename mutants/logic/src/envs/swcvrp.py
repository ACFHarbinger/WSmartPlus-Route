"""
SCWCVRP Environment implementation.

Stochastic Capacitated Waste Collection Vehicle Routing Problem:
Agent sees noisy estimates of bin fill levels but collects real waste.
Rewards are based on real waste levels and overflows.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from logic.src.envs.generators import SCWCVRPGenerator
from logic.src.envs.wcvrp import WCVRPEnv
from tensordict import TensorDict


class SCWCVRPEnv(WCVRPEnv):
    """
    Stochastic Capacitated Waste Collection VRP Environment.

    The agent must collect waste from bins based on noisy observations.
    The real waste level is used for reward calculation (overflows).
    """

    name: str = "scwcvrp"

    def __init__(
        self,
        generator: Optional[SCWCVRPGenerator] = None,
        generator_params: Optional[dict] = None,
        overflow_penalty: float = 10.0,
        collection_reward: float = 1.0,
        cost_weight: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """Initialize SCWCVRPEnv."""
        generator_params = generator_params or kwargs
        if generator is None:
            generator = SCWCVRPGenerator(**generator_params, device=device)

        super().__init__(
            generator=generator,
            generator_params=generator_params,
            overflow_penalty=overflow_penalty,
            collection_reward=collection_reward,
            cost_weight=cost_weight,
            device=device,
            **kwargs,
        )

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize SCWCVRP episode state."""
        td = super()._reset_instance(td)

        # Ensure real_waste is tracked
        if "real_waste" not in td.keys():
            # If not provided by generator, assume it's same as demand (no noise)
            td["real_waste"] = td["demand"].clone()

        # Track total real waste collected
        td["total_real_collected"] = torch.zeros_like(td["total_collected"])

        return td

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Execute action and update state with real waste collection."""
        action = td["action"]
        is_not_depot = action != 0

        # Collection logic for real waste
        real_waste_at_node = td["real_waste"].gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Update real waste tracking (internal)
        # We need to ensure we don't collect more than vehicle capacity
        # WCVRPEnv base class already handles current_load and current_node update
        # but it uses 'demand' (which is the noisy version here).
        # However, for reward we want to know what was *actually* collected.

        # We first call super to get the basic state updates
        # Wait, if I call super it will use td['demand'] to update current_load and total_collected.
        # But in SCWCVRP, the agent thinks it collects 'demand' (noisy),
        # but it actually collects based on capacity.

        # In legacy logic:
        # actual_collected = min(real_waste, remaining_capacity)
        # current_node_wastes -= actual_collected

        remaining_cap = td["capacity"] - td["current_load"]
        actual_collected = torch.min(real_waste_at_node, remaining_cap)
        actual_collected = actual_collected * is_not_depot.float()

        # Update real waste in the bins
        new_real_waste = td["real_waste"].clone()
        new_real_waste.scatter_(
            1, action.unsqueeze(-1), td["real_waste"].gather(1, action.unsqueeze(-1)) - actual_collected.unsqueeze(-1)
        )
        td["real_waste"] = new_real_waste

        # Now update noisy demand - usually setting to 0 after visit if we think we cleared it
        # or update based on what we think we collected?
        # Standard WCVRPEnv sets td['demand'] to 0 at the node.

        td = super()._step_instance(td)

        # Override the collection tracking with real values if desired,
        # or just track total_real_collected.
        td["total_real_collected"] = td["total_real_collected"] + actual_collected

        return td

    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute SCWCVRP reward based on REAL waste levels and collection.
        """
        # We use total_real_collected instead of total_collected
        collection = td["total_real_collected"]
        cost = td["tour_length"]

        # Add return to depot
        current = td["current_node"].squeeze(-1)
        locs = td["locs"]
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        depot_loc = td["depot"]
        return_distance = torch.norm(depot_loc - current_loc, dim=-1)

        not_at_depot = current != 0
        total_cost = cost + return_distance * not_at_depot.float()

        # Overflows based on remaining REAL waste
        max_waste = td.get("max_waste", torch.tensor(1.0, device=td.device))
        if max_waste.dim() > 0:
            max_waste = max_waste.unsqueeze(-1)

        # In legacy, overflow is uncollected real waste >= max_waste
        overflows = (td["real_waste"][..., 1:] >= max_waste).float().sum(-1)

        # Store for logging
        td["real_collection"] = collection
        td["cost"] = total_cost
        td["real_overflows"] = overflows

        # Store for GDPO
        td["reward_collection"] = collection
        td["reward_cost"] = -total_cost
        td["reward_overflow"] = -overflows

        reward = self.collection_reward * collection - self.cost_weight * total_cost - self.overflow_penalty * overflows
        return reward
