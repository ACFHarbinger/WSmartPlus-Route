"""
TSPkopt Environment implementation.

Improvement-based Traveling Salesman Problem (TSP) supporting iterative
local search moves (e.g. 2-opt).
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base import ImprovementEnvBase
from logic.src.envs.generators import TSPGenerator


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
        d = locs.gather(1, solution.unsqueeze(-1).expand(-1, -1, 2))

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
