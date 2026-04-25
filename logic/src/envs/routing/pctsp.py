"""
PCTSP Environment — Prize-Collecting Traveling Salesman Problem.

The agent collects node prizes while avoiding penalties for skipping nodes.
A minimum total prize must be collected before the agent can return to the
depot.  The objective balances saved penalties against travel cost.

Reward: saved_penalties - (tour_length + total_remaining_penalties)
Done:   agent returns to depot after collecting at least prize_required
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.generators.pctsp import PCTSPGenerator


class PCTSPEnv(RL4COEnvBase):
    """
    Prize-Collecting Traveling Salesman Problem (PCTSP) environment.

    Node layout in ``td["locs"]``:
      index 0 = depot, indices 1…N = customers.

    In the deterministic variant (PCTSP) the ``real_prize`` revealed upon
    visiting a node equals the ``deterministic_prize`` known up front.
    In the stochastic variant (SPCTSP) the ``stochastic_prize`` is used.
    """

    NAME: str = "pctsp"
    name: str = "pctsp"
    node_dim: int = 2
    _stochastic: bool = False

    def __init__(
        self,
        generator: Optional[PCTSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the PCTSP environment."""
        generator_params = generator_params or {}
        if generator is None:
            generator = PCTSPGenerator(**generator_params, device=device)
        super().__init__(generator, generator_params, device, **kwargs)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize PCTSP episode state."""
        if self.generator is None:
            raise ValueError(f"Generator for {self.NAME} is not initialized. Initialize with an instance first.")
        if "visited" in td.keys():
            return td

        device = td.device
        bs = td.batch_size

        # Prepend depot to locs when customer-only coords are provided
        locs = td["locs"]
        gen_n = getattr(self.generator, "num_loc", None)
        if "depot" in td.keys() and (gen_n is None or locs.shape[-2] == gen_n):
            td["locs"] = torch.cat([td["depot"].unsqueeze(-2), locs], dim=-2)

        num_nodes = td["locs"].shape[-2]  # N+1

        # Real prize: stochastic (SPCTSP) or deterministic (PCTSP)
        real_prize_cust = td["stochastic_prize"] if self._stochastic else td["deterministic_prize"]
        # Prepend 0 for depot
        td["real_prize"] = torch.cat([torch.zeros(*bs, 1, device=device), real_prize_cust], dim=-1)  # [B, N+1]

        # Penalty: prepend 0 for depot
        td["penalty"] = torch.cat([torch.zeros(*bs, 1, device=device), td["penalty"]], dim=-1)  # [B, N+1]

        # Expected prize is the deterministic prize (for policy conditioning)
        td["expected_prize"] = td["deterministic_prize"]  # [B, N]

        td["current_node"] = torch.zeros(*bs, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["cur_total_prize"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        # Total penalty = sum of all customer penalties (they're "owed" until visited)
        td["cur_total_penalty"] = td["penalty"][..., 1:].sum(-1)  # [B]

        if self.generator is None:
            raise ValueError("Generator must be initialized for PCTSP environment.")
        td["prize_required"] = torch.full((*bs,), self.generator.prize_required, device=device, dtype=torch.float32)

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
        Execute one PCTSP action.

        When visiting a customer node, the real prize is collected and the
        node's penalty is removed from the running total (it is "saved").
        """
        action = td["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        current = td["current_node"]
        if current.dim() > 1:
            current = current.squeeze(-1)

        locs = td["locs"]
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        td["tour_length"] = td["tour_length"] + distance

        # Collect real prize and remove penalty for visited node
        prize_here = td["real_prize"].gather(-1, action.unsqueeze(-1)).squeeze(-1)
        penalty_here = td["penalty"].gather(-1, action.unsqueeze(-1)).squeeze(-1)

        td["cur_total_prize"] = td["cur_total_prize"] + prize_here
        td["cur_total_penalty"] = td["cur_total_penalty"] - penalty_here

        td["visited"] = td["visited"].scatter(1, action.unsqueeze(-1), True)
        td["current_node"] = action
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    # ------------------------------------------------------------------
    # Done / Mask / Reward
    # ------------------------------------------------------------------

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Done when agent returns to depot after at least one step."""
        current = td["current_node"]
        if current.dim() > 1:
            current = current.squeeze(-1)
        step = td["i"].squeeze(-1) if td["i"].dim() > 1 else td["i"]
        return (current == 0) & (step > 0)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Depot (index 0) is blocked until the agent has collected at least
        ``prize_required`` total prize, unless all customers are visited.

        All already-visited nodes are masked out.
        """
        visited = td["visited"]  # [B, N+1]
        # Also mask out nodes reachable from an already-done episode (depot visited)
        depot_done = visited[..., 0:1]  # [B, 1]
        mask = ~(visited | depot_done)  # [B, N+1]

        # Depot: only open when prize_required is met (or all customers visited)
        prize_met = td["cur_total_prize"] >= td["prize_required"]
        all_visited = visited[..., 1:].all(dim=-1)
        depot_open = prize_met | all_visited
        mask[..., 0] = depot_open

        return mask

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
                Compute reward for PCTSP: Prize - Cost - Penalty.
        ies - total tour cost.

                ``saved_penalties`` are the penalties of visited customer nodes.
                ``total_tour_cost`` = tour_length + penalties of all unvisited nodes
                (still "owed").
        """
        penalty = td["penalty"]  # [B, N+1] with depot=0

        if actions is not None and actions.size(-1) > 0:
            # Gather penalties of visited customers
            saved_pen = penalty.gather(1, actions).sum(-1)
            # Remaining penalties (all customer penalties - saved)
            remaining_pen = penalty[..., 1:].sum(-1) - saved_pen
        else:
            saved_pen = penalty[..., 1:].sum(-1) - td["cur_total_penalty"]
            remaining_pen = td["cur_total_penalty"]

        return saved_pen - (td["tour_length"] + remaining_pen)
