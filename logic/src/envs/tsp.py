"""
TSP Environment implementation.

Standard Traveling Salesman Problem (TSP) and its improvement-based
variant (TSPkopt) supporting iterative local search moves.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base import ImprovementEnvBase, RL4COEnvBase
from logic.src.envs.generators import TSPGenerator


class TSPEnv(RL4COEnvBase):
    """
    Traveling Salesman Problem Environment.
    The agent must visit all nodes exactly once and return to the depot
    while minimizing the total distance traveled.
    """

    NAME = "tsp"
    name: str = "tsp"

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
        return all_visited & current_is_depot


class TSPkoptEnv(ImprovementEnvBase):
    """
    TSP Environment for improvement-based methods (k-opt).

    State contains a 'solution' field (tour) which is modified by actions.
    Actions are typically pairs of indices to swap or reverse sections.
    """

    name: str = "tsp_kopt"

    def __init__(
        self,
        generator: Optional[TSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """Initialize TSPkoptEnv."""
        generator_params = generator_params or kwargs
        if generator is None:
            generator = TSPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """For improvement moves, all nodes are typically valid targets.
        Return a mask that allows all nodes.
        """
        bs = td.batch_size
        num_nodes = td["locs"].shape[-2] + 1
        return torch.ones(*bs, num_nodes, dtype=torch.bool, device=self.device)

    def _get_initial_solution(self, td: TensorDict) -> torch.Tensor:
        """Generate random initial tour starting at depot."""
        bs = td.batch_size[0]
        num_nodes = td["locs"].shape[-2] + 1

        # Shuffle nodes 1 to N
        tour = torch.stack([torch.randperm(num_nodes - 1, device=self.device) + 1 for _ in range(bs)])

        # Prepend depot (0)
        depot = torch.zeros(bs, 1, dtype=torch.long, device=self.device)
        full_tour = torch.cat([depot, tour], dim=1)

        return full_tour

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Apply k-opt move to the solution.
        Action: (idx1, idx2) - edges to swap for 2-opt.
        """
        action = td["action"]  # [batch, 2] or combined
        solution = td["solution"]

        # 2-opt implementation: reverse section between i and j
        # Assuming action contains i and j
        if action.dim() == 1:  # Single int encoding if necessary
            # Convert if needed
            pass

        i, j = action[:, 0], action[:, 1]

        # Ensure i < j
        i, j = torch.min(i, j), torch.max(i, j)

        # Apply 2-opt reversal
        new_solution = solution.clone()
        for b in range(solution.size(0)):
            # slice [i+1:j+1] reversed
            idx_i, idx_j = i[b].item(), j[b].item()
            if idx_i < idx_j:
                new_solution[b, idx_i + 1 : idx_j + 1] = solution[b, idx_i + 1 : idx_j + 1].flip(0)

        td["solution"] = new_solution
        return td

    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return negative tour length of current solution."""
        solution = td["solution"]

        depot = td["depot"]
        customers = td["locs"]
        locs = torch.cat([depot.unsqueeze(1), customers], dim=1)

        # Gather coordinates in tour order
        d = locs.gather(1, solution[:, None].expand(*solution.size(), 2))

        # Total distance: sum segments + closed loop
        length = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1) + (d[:, -1] - d[:, 0]).norm(p=2, dim=-1)

        return -length

    def _get_initial_reward(self, td: TensorDict) -> torch.Tensor:
        """Compute reward for initial solution."""
        return self._get_reward(td)

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Usually improvement continues for fixed steps or until converge.
        Here we define a max_steps entry in td if present.
        """
        if "max_steps" in td.keys():
            return td["i"] >= td["max_steps"]
        return torch.zeros(td.batch_size, dtype=torch.bool, device=self.device)
