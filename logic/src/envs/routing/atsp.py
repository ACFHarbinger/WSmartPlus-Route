"""
ATSP Environment — Asymmetric Traveling Salesman Problem.

The agent must visit all N nodes exactly once and return to the starting
node, minimising the total asymmetric tour cost read from a cost matrix.
Unlike the symmetric TSP there is no depot; the agent picks its own start.

Reward:  -(total tour cost including the closing edge last→first)
Done:    all N nodes have been visited

Attributes:
    ATSPEnv: RL4CO environment for the Asymmetric Traveling Salesman Problem.

Example:
    >>> from logic.src.envs.routing import ATSPEnv
    >>> env = ATSPEnv()
    >>> td = env.reset(batch_size=[4])
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.generators.atsp import ATSPGenerator


class ATSPEnv(RL4COEnvBase):
    """
    Asymmetric Traveling Salesman Problem (ATSP) environment.

    The environment uses an asymmetric cost matrix rather than spatial
    coordinates.  No depot exists: the agent starts at whichever node it
    selects first and must return there at the end (closing edge handled
    in reward computation).

    Attributes:
        NAME: Environment identifier string ``"atsp"``.
        node_dim: Node feature dimension (0 — no spatial coordinates).
    """

    NAME: str = "atsp"
    name: str = "atsp"
    node_dim: int = 0  # no spatial coordinates; uses cost_matrix

    def __init__(
        self,
        generator: Optional[ATSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the ATSP environment.

        Args:
            generator: Pre-built ATSPGenerator; created from generator_params if None.
            generator_params: Keyword arguments forwarded to ATSPGenerator constructor.
            device: Torch device for tensor placement.
            kwargs: Additional arguments forwarded to RL4COEnvBase.
        """
        generator_params = generator_params or {}
        if generator is None:
            generator = ATSPGenerator(**generator_params, device=device)
        super().__init__(generator, generator_params, device, **kwargs)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialise ATSP episode state.

        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        if "visited" in td.keys():
            return td

        device = td.device
        bs = td.batch_size
        if self.generator is None:
            raise ValueError("Generator is not initialized.")
        n = td["cost_matrix"].shape[-1]

        td["current_node"] = torch.zeros(*bs, dtype=torch.long, device=device)
        td["first_node"] = torch.zeros(*bs, dtype=torch.long, device=device)
        # All nodes start unvisited; node 0 is marked as starting point
        td["visited"] = torch.zeros(*bs, n, dtype=torch.bool, device=device)
        td["visited"][..., 0] = True

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
        """Delegate to OpsMixin._step, which calls _step_instance.

        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        return OpsMixin._step(self, td)

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Execute one ATSP action.

        Uses the asymmetric cost matrix to look up the travel cost from
        the current node to the selected action node.
        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        action = td["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        current_node = td["current_node"]
        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)
        if current_node.dim() == 0:
            current_node = current_node.unsqueeze(0)

        # Batch-safe cost matrix lookup: cost[b, current[b], action[b]]
        cost_matrix = td["cost_matrix"]  # [*B, N, N]
        bs = td.batch_size
        flat_bs = bs.numel() if len(bs) > 0 else 1
        flat_cost = cost_matrix.reshape(flat_bs, cost_matrix.shape[-2], cost_matrix.shape[-1])
        batch_idx = torch.arange(flat_bs, device=td.device)
        dist = flat_cost[batch_idx, current_node.reshape(flat_bs), action.reshape(flat_bs)].reshape(*bs)

        td["tour_length"] = td["tour_length"] + dist
        td["visited"] = td["visited"].scatter(-1, action.unsqueeze(-1), True)
        td["current_node"] = action
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    # ------------------------------------------------------------------
    # Done / Mask / Reward
    # ------------------------------------------------------------------

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Done when all N nodes have been visited.

        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        return td["visited"].all(dim=-1)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """All unvisited nodes are valid actions.

        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        return ~td["visited"]

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reward = -(accumulated tour cost + closing edge cost).

        The closing edge goes from the last visited node back to the first
        node selected (stored in ``td["first_node"]``).
        Args:
            td: Input TensorDict containing the environment state.
            actions: Tensor of actions to take.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        cost_matrix = td["cost_matrix"]
        bs = td.batch_size
        flat_bs = bs.numel() if len(bs) > 0 else 1
        flat_cost = cost_matrix.reshape(flat_bs, cost_matrix.shape[-2], cost_matrix.shape[-1])

        # Closing edge: current_node → first_node
        current = td["current_node"]
        if current.dim() > 1:
            current = current.squeeze(-1)
        first = td["first_node"]
        if first.dim() > 1:
            first = first.squeeze(-1)

        batch_idx = torch.arange(flat_bs, device=td.device)
        closing = flat_cost[batch_idx, current.reshape(flat_bs), first.reshape(flat_bs)].reshape(*bs)

        total = td["tour_length"] + closing
        return -total
