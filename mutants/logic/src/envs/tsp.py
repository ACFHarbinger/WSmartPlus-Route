"""
TSP Environment implementation.

Standard Traveling Salesman Problem (TSP) where the agent must visit all nodes
exactly once and return to the depot while minimizing the total distance traveled.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from logic.src.envs.base import RL4COEnvBase
from logic.src.envs.generators import TSPGenerator
from tensordict import TensorDict


class TSPEnv(RL4COEnvBase):
    """
    Traveling Salesman Problem Environment.
    The agent must visit all nodes exactly once and return to the depot
    while minimizing the total distance traveled.
    """

    NAME = "tsp"
    name: str = "tsp"
    node_dim: int = 2  # (x, y)

    def __init__(
        self,
        generator: Optional[TSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """Initialize TSPEnv."""
        generator_params = generator_params or kwargs
        if generator is None:
            generator = TSPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize TSP state."""
        device = td.device
        bs = td.batch_size
        num_nodes = td["locs"].shape[-2] + 1  # Customers + Depot

        # Current node (start at depot at index 0)
        td["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["visited"].index_fill_(-1, torch.tensor([0], device=device), True)  # Depot is visited initially

        # Tour tracking
        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, device=device)
        td["first_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)

        # RL4CO/TorchRL expected keys
        td["reward"] = torch.zeros(*bs, device=device)
        td["terminated"] = torch.zeros(*bs, dtype=torch.bool, device=device)
        td["truncated"] = torch.zeros(*bs, dtype=torch.bool, device=device)

        return td

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Execute action and update state."""
        action = td["action"]
        current = td["current_node"].squeeze(-1)

        # Combine depot and customers for full locs
        depot = td["depot"]
        customers = td["locs"]
        locs = torch.cat([depot.unsqueeze(1), customers], dim=1)

        # Compute distance traveled
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        # Update tour length
        td["tour_length"] = td["tour_length"] + distance

        # Update visited
        td["visited"] = td["visited"].scatter(1, action.unsqueeze(-1), True)

        # Update current node
        td["current_node"] = action.unsqueeze(-1)

        # Append to tour
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        # print(f"DEBUG: Step i={td['i'].item()}, action={action.item()}, visited={td['visited'].sum().item()}")

        return td

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Can visit any unvisited node. Return to depot only when all nodes visited."""
        mask = ~td["visited"].clone()

        # If all nodes visited (except depot), depot is the only valid action
        all_visited = td["visited"][:, 1:].all(dim=-1)

        # If not all visited, depot is invalid
        mask[~all_visited, 0] = False

        # Logic adjustment for batch:
        mask[:, 0] = all_visited

        return mask

    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reward is negative total tour length."""
        # Add return to depot if needed
        # (This is handled by last step in autoregressive construction)
        return -td["tour_length"]

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Done when all nodes visited and back at depot."""
        all_visited = td["visited"].all(dim=-1)
        current_is_depot = td["current_node"].squeeze(-1) == 0
        done = all_visited & current_is_depot

        # Debugging infinite loop
        # if done.any(): print(f"DEBUG: TSP Done at nodes {td['visited'].sum(-1)}")

        return done
