"""
PDP Environment — Pickup and Delivery Problem.

The agent must visit all pickup nodes before their corresponding delivery
nodes, minimising total travel distance.  The depot is not visited during
the episode; it is added implicitly in the reward computation.

Node layout (locs, index 0 = depot):
  [0]              — depot (not visited during episode)
  [1  …  K]        — K pickup nodes
  [K+1 …  2K]      — K delivery nodes  (delivery K+i pairs with pickup i)
where K = num_loc / 2.

Reward: -(total tour length, depot → tour → depot)
Done:   all pickup/delivery nodes visited
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.generators.pdp import PDPGenerator


class PDPEnv(RL4COEnvBase):
    """
    Pickup and Delivery Problem (PDP) environment.

    Ordering constraint: pickup node i must be visited before its paired
    delivery node i + K.  The ``to_deliver`` mask tracks which delivery
    nodes have had their pickup completed and are therefore unlocked.
    """

    NAME: str = "pdp"
    name: str = "pdp"
    node_dim: int = 2

    def __init__(
        self,
        generator: Optional[PDPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the PDP environment."""
        generator_params = generator_params or {}
        if generator is None:
            generator = PDPGenerator(**generator_params, device=device)
        super().__init__(generator, generator_params, device, **kwargs)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """
        Initialize PDP episode state.

        Args:
            td (TensorDict): Input tensor dictionary containing:
                - locs (Tensor): Node coordinates, shape (batch, num_loc, 2)
                - depot (Tensor): Depot coordinates, shape (batch, 2)
                - visited (Tensor): Visited mask, shape (batch, num_loc + 1)
                - current_node (LongTensor): Current position index, shape (batch, 1)
                - i (LongTensor): Current step, shape (batch, 1)

        Returns:
            TensorDict: Updated tensor dictionary with initialized state.
        """
        if self.generator is None:
            raise ValueError(f"Generator for {self.NAME} is not initialized. Initialize with an instance first.")
        if "visited" in td.keys():
            return td

        device = td.device
        bs = td.batch_size
        num_loc = self.generator.num_loc  # total customer nodes (even)
        k = num_loc // 2  # number of pickup/delivery pairs

        # Prepend depot to locs
        locs = td["locs"]
        gen_n = getattr(self.generator, "num_loc", None)
        if "depot" in td.keys() and (gen_n is None or locs.shape[-2] == gen_n):
            td["locs"] = torch.cat([td["depot"].unsqueeze(-2), locs], dim=-2)

        # to_deliver[b, n]: True = node n is currently eligible to visit
        # Initially only pickup nodes (indices 1..k) and depot are eligible;
        # delivery nodes (k+1..2k) are not until their pickup is done.
        to_deliver = torch.cat(
            [
                torch.ones(*bs, k + 1, dtype=torch.bool, device=device),  # depot + pickups
                torch.zeros(*bs, k, dtype=torch.bool, device=device),  # deliveries (locked)
            ],
            dim=-1,
        )  # [B, 2k+1]

        # available: all customer nodes are available; depot is NOT visited
        # (depot is handled implicitly in reward; agent never visits it during episode)
        available = torch.ones(*bs, num_loc + 1, dtype=torch.bool, device=device)
        available[..., 0] = False  # depot excluded from episode actions

        td["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        td["visited"] = ~available  # depot starts as "visited" (unavailable)
        td["to_deliver"] = to_deliver
        td["available"] = available

        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, dtype=torch.float32, device=device)

        td["reward"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["terminated"] = torch.zeros(*bs, dtype=torch.bool, device=device)
        td["truncated"] = torch.zeros(*bs, dtype=torch.bool, device=device)

        return td

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step(self, td: TensorDict) -> TensorDict:
        return OpsMixin._step(self, td)

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Execute one PDP action.

        After visiting a pickup node i, unlocks its delivery node i + K.

        Args:
            td (TensorDict): Input tensor dictionary containing:
                - action (LongTensor): Action to take, shape (batch, 1)
                - current_node (LongTensor): Current position index, shape (batch, 1)
                - locs (Tensor): Node coordinates, shape (batch, num_loc, 2)
                - available (Tensor): Availability mask, shape (batch, num_loc + 1)
                - to_deliver (Tensor): Delivery eligibility mask, shape (batch, num_loc + 1)
                - tour (Tensor): Current tour, shape (batch, max_steps)
                - tour_length (Tensor): Current tour length, shape (batch,)

        Returns:
            TensorDict: Updated tensor dictionary with new state.
        """
        if self.generator is None:
            raise ValueError("Generator must be initialized for PDP environment.")

        action = td["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        num_loc = self.generator.num_loc
        k = num_loc // 2

        current = td["current_node"].squeeze(-1)
        locs = td["locs"]
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        td["tour_length"] = td["tour_length"] + distance

        # Mark node as visited / unavailable
        available = td["available"].scatter(-1, action.unsqueeze(-1), False)
        visited = td["visited"].scatter(-1, action.unsqueeze(-1), True)

        # Unlock paired delivery when a pickup is visited:
        # pickup i (index 1..k) → unlock delivery at index i + k
        new_to_deliver = (action + k) % (num_loc + 1)  # [B]
        to_deliver = td["to_deliver"].scatter(-1, new_to_deliver.unsqueeze(-1), True)

        td["available"] = available
        td["visited"] = visited
        td["to_deliver"] = to_deliver
        td["current_node"] = action.unsqueeze(-1)
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    # ------------------------------------------------------------------
    # Done / Mask / Reward
    # ------------------------------------------------------------------

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Done when no customer nodes remain available."""
        return ~td["available"][..., 1:].any(dim=-1)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Valid actions: nodes that are both available AND to_deliver.
        Depot (index 0) is never a valid action.
        """
        return td["available"] & td["to_deliver"]

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reward = -(tour length including depot at start and end).

        The tour stored in ``td["tour"]`` does not include the depot;
        the return-to-depot distance is added here.
        """
        locs = td["locs"]
        depot_loc = locs[..., 0:1, :]  # [B, 1, 2]

        # Current position (last node visited)
        current = td["current_node"].squeeze(-1)
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)

        # Return to depot distance
        return_dist = torch.norm(current_loc - depot_loc.squeeze(1), dim=-1)
        total = td["tour_length"] + return_dist

        return -total
