"""
OP Environment — Orienteering Problem.

The agent selects a subset of customers to visit within a given distance
budget, maximising the total collected prize.  The tour must start and
end at the depot; not all customers need to be visited.

Reward: sum of prizes collected
Done:   agent returns to depot (action = 0, step > 0)

Attributes:
    OPEnv: OP environment.

Example:
    >>> from logic.src.envs.routing import get_env
    >>> env = get_env("op", num_loc=50)
    >>> td = env.reset()
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.generators.op import OPGenerator


class OPEnv(RL4COEnvBase):
    """
    Orienteering Problem (OP) environment.

    Node layout in ``td["locs"]``:
      index 0 = depot, indices 1…N = customers (each with a prize).

    The episode ends when the agent voluntarily returns to the depot.
    The ``max_length`` budget is enforced via the action mask: a customer
    is only selectable if the agent can reach it AND return to the depot
    within the remaining budget.

    Attributes:
        NAME: Name of the environment.
        name: Name of the environment.
        node_dim: Node feature dimension (x, y coordinates only → 2).
    """

    NAME: str = "op"
    name: str = "op"
    node_dim: int = 2

    def __init__(
        self,
        generator: Optional[OPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the OP environment.

        Args:
            generator: Pre-built OPGenerator instance.  A new one is created
                from *generator_params* if not supplied.
            generator_params: Keyword arguments forwarded to OPGenerator when
                *generator* is None.
            device: Torch device string or object.
            kwargs: Additional keyword arguments forwarded to the base class.

        Returns:
            None
        """
        generator_params = generator_params or {}
        if generator is None:
            generator = OPGenerator(**generator_params, device=device)
        super().__init__(generator, generator_params, device, **kwargs)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialise OP episode state.

        Args:
            td: Input TensorDict containing graph structure and node properties.

        Returns:
            TensorDict: Initialized OP state with per-episode mutable fields.
        """
        if "visited" in td.keys():
            return td

        device = td.device
        bs = td.batch_size

        # Prepend depot to locs when customer-only coords are provided
        locs = td["locs"]
        if self.generator is None:
            raise ValueError("Generator is not initialized.")
        gen_n = getattr(self.generator, "num_loc", None)
        if "depot" in td.keys() and (gen_n is None or locs.shape[-2] == gen_n):
            td["locs"] = torch.cat([td["depot"].unsqueeze(-2), locs], dim=-2)

        locs_all = td["locs"]  # [B, N+1, 2]
        num_nodes = locs_all.shape[-2]

        # Prepend zero prize for depot
        prize = td["prize"]  # [B, N]
        td["prize"] = torch.cat([torch.zeros(*bs, 1, device=device), prize], dim=-1)  # [B, N+1]

        # max_length is the total budget.  Internally we track remaining budget
        # adjusted by the return distance from each candidate node to the depot,
        # so the agent cannot overshoot.
        max_length = td["max_length"]  # [B]
        if max_length.dim() > 1:
            max_length = max_length.squeeze(-1)
        # Per-node remaining budget when standing at depot:
        # max_length_at_node[n] = max_length - dist(n, depot)  (agent must be able to return)
        dist_to_depot = (locs_all[..., 0:1, :] - locs_all).norm(p=2, dim=-1)  # [B, N+1]
        td["max_length"] = max_length.unsqueeze(-1) - dist_to_depot - 1e-6  # [B, N+1]

        td["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["tour_length"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["current_total_prize"] = torch.zeros(*bs, dtype=torch.float32, device=device)

        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["reward"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["terminated"] = torch.zeros(*bs, dtype=torch.bool, device=device)
        td["truncated"] = torch.zeros(*bs, dtype=torch.bool, device=device)

        return td

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step(self, td: TensorDict) -> TensorDict:
        """Execute one step in the environment.

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            TensorDict: Updated TensorDict with new state and reward information.
        """
        return OpsMixin._step(self, td)

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Execute one OP routing action: move, collect prize, update length.

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            TensorDict: Updated TensorDict with new state and reward information.
        """
        action = td["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        current = td["current_node"].squeeze(-1)

        locs = td["locs"]
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        td["tour_length"] = td["tour_length"] + distance

        # Collect prize from this node (depot prize = 0, already padded)
        prize_here = td["prize"].gather(-1, action.unsqueeze(-1)).squeeze(-1)
        td["current_total_prize"] = td["current_total_prize"] + prize_here

        td["visited"] = td["visited"].scatter(1, action.unsqueeze(-1), True)
        td["current_node"] = action.unsqueeze(-1)
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    # ------------------------------------------------------------------
    # Done / Mask / Reward
    # ------------------------------------------------------------------

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Done when the agent returns to the depot after at least one move.

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            BoolTensor [*B] — True for completed episodes.
        """
        current = td["current_node"].squeeze(-1)
        step = td["i"].squeeze(-1) if td["i"].dim() > 1 else td["i"]
        return (current == 0) & (step > 0)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        A customer is valid if:
          (a) not yet visited,
          (b) not in a done episode (depot already visited),
          (c) can be reached and depot reached after without exceeding budget.

        Depot (index 0) is always valid.

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            BoolTensor [*B, num_nodes] — True indicates a selectable action.
        """
        current = td["current_node"].squeeze(-1)
        current_loc = td["locs"].gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)

        # Distance from current position to each node
        dist_current_to_node = (td["locs"] - current_loc.unsqueeze(1)).norm(p=2, dim=-1)  # [B, N+1]

        # Budget check: tour_length + dist(current→node) <= max_length[node]
        exceeds_budget = td["tour_length"].unsqueeze(-1) + dist_current_to_node > td["max_length"]

        # Invalid: already visited OR visited depot (done) OR exceeds budget
        depot_visited = td["visited"][..., 0:1]  # [B, 1]
        mask = ~(td["visited"] | depot_visited | exceeds_budget)

        # Depot always valid
        mask[..., 0] = True
        return mask

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reward is the total collected prize.

        Args:
            td: Input TensorDict containing action and state information.
            actions: Action sequence (unused in this implementation).

        Returns:
            Tensor [*B] — scalar reward per batch instance.
        """
        return td["current_total_prize"]
