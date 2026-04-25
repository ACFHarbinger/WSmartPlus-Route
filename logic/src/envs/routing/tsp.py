"""
TSP Environment — Traveling Salesman Problem.

The agent must visit all nodes exactly once and return to the depot
while minimising the total distance traveled.  The depot is prepended
to the customer locations so that index 0 always refers to the depot.

Reward: -(total tour length)
Done:   all nodes visited and agent back at depot

Attributes:
    TSPEnv: RL4CO environment for the Traveling Salesman Problem.

Example:
    >>> from logic.src.envs.routing import get_env
    >>> env = get_env("tsp", num_loc=20)
    >>> td = env.reset(batch_size=[4])
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.generators import TSPGenerator


class TSPEnv(RL4COEnvBase):
    """
    Traveling Salesman Problem (TSP) environment.

    The depot (index 0) is the starting and finishing city.  All other
    nodes are customers that must be visited exactly once.  The policy
    selects the next unvisited customer at each step; when all customers
    have been visited, only the depot is a valid action.

    Attributes:
        NAME: Environment identifier string ``"tsp"``.
        name: Alias for NAME used by the registry.
        node_dim: Node feature dimension — 2 for (x, y) coordinates.
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
        """Initialize TSPEnv.

        Args:
            generator: Pre-built TSPGenerator; created from generator_params if None.
            generator_params: Keyword arguments forwarded to TSPGenerator constructor.
            device: Torch device for tensor placement.
            kwargs: Additional arguments forwarded to RL4COEnvBase.
        """
        generator_params = generator_params or kwargs
        if generator is None:
            generator = TSPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize TSP state.

        Args:
            td: TensorDict produced by the generator, containing ``locs``
                (customer coordinates) and ``depot`` tensors.

        Returns:
            TensorDict: Initialised state with ``current_node``, ``visited``,
                ``tour``, ``tour_length``, ``first_node``, ``reward``,
                ``terminated``, and ``truncated`` fields.
        """
        if self.generator is None:
            raise ValueError(f"Generator for {self.NAME} is not initialized. Initialize with an instance first.")
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
        """Execute action and update state.

        Args:
            td: Current TensorDict state containing ``action``, ``current_node``,
                ``depot``, ``locs``, ``visited``, ``tour``, and ``tour_length``.

        Returns:
            TensorDict: Updated state with incremented ``tour_length``, updated
                ``visited`` mask, ``current_node`` set to the selected action,
                and the action appended to ``tour``.
        """
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
        """Can visit any unvisited node. Return to depot only when all nodes visited.

        Args:
            td: Current state containing ``visited`` and ``current_node`` fields.

        Returns:
            torch.Tensor: Boolean mask of shape ``(batch, num_nodes + 1)`` where
                ``True`` marks a valid action.  Depot (index 0) is valid only
                when all customers have been visited.
        """
        mask = ~td["visited"].clone()

        # If all nodes visited (except depot), depot is the only valid action
        all_visited = td["visited"][:, 1:].all(dim=-1)

        # If not all visited, depot is invalid
        mask[~all_visited, 0] = False

        # Logic adjustment for batch:
        mask[:, 0] = all_visited

        return mask

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reward is negative total tour length.

        Args:
            td: Final state TensorDict containing ``tour_length``.
            actions: Unused; reward is derived from the accumulated ``tour_length``.

        Returns:
            torch.Tensor: Scalar reward per batch element, equal to
                ``-tour_length``, shape ``(batch,)``.
        """
        # actions is unused here as reward is state-based (tour_length)
        return -td["tour_length"]

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Done when all nodes visited and back at depot.

        Args:
            td: Current state containing ``visited`` (all nodes) and
                ``current_node`` tensors.

        Returns:
            torch.Tensor: Boolean tensor of shape ``(batch,)``; ``True`` when
                all nodes have been visited and the agent is back at the depot.
        """
        all_visited = td["visited"].all(dim=-1)
        current_is_depot = td["current_node"].squeeze(-1) == 0
        done = all_visited & current_is_depot
        return done
