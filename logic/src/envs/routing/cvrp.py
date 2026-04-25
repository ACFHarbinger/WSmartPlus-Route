"""
CVRP Environment — Capacitated Vehicle Routing Problem.

The agent routes a single vehicle with finite capacity from a depot to
serve all customers, returning to the depot whenever the load is
exhausted.  The objective is to minimise total travel distance.

Reward: -(total tour length)
Done:   all customers visited and vehicle back at depot
Attributes:
    CVRPEnv: Capacitated Vehicle Routing Problem (CVRP) environment.

Example:
    >>> import torch
    >>> from logic.src.envs.routing.cvrp import CVRPEnv
    >>> env = CVRPEnv(batch_size=4)
    >>> env.reset()
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.generators.cvrp import CVRPGenerator


class CVRPEnv(RL4COEnvBase):
    """
    Capacitated Vehicle Routing Problem (CVRP) environment.

    Node layout in ``td["locs"]``:
      index 0 = depot, indices 1…N = customers.

    The vehicle starts at the depot, visits customers, and can return to
    the depot (index 0) at any time to refill.  The episode ends when all
    customers have been served and the vehicle has returned to the depot.

    Attributes:
        NAME: Environment name identifier.
        name: Environment name identifier.
        node_dim: Dimension of the node coordinates.
    """

    NAME: str = "cvrp"
    name: str = "cvrp"
    node_dim: int = 2

    def __init__(
        self,
        generator: Optional[CVRPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the CVRP environment.

        Args:
            generator: Data generator for the environment.
            generator_params: Parameters for the data generator.
            device: Torch device to place tensors on.
            kwargs: Additional keyword arguments.
        """
        generator_params = generator_params or {}
        if generator is None:
            generator = CVRPGenerator(**generator_params, device=device)
        super().__init__(generator, generator_params, device, **kwargs)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize CVRP episode state.

        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
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

        num_nodes = td["locs"].shape[-2]  # num_loc + 1

        td["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        # visited: all zeros — depot gets marked when agent explicitly returns there
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["used_capacity"] = torch.zeros(*bs, dtype=torch.float32, device=device)

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
        Execute one CVRP routing action.

        Computes Euclidean travel distance, marks the node as visited,
        and updates the vehicle's remaining capacity.  Visiting the depot
        (action = 0) resets the on-board load.
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

        current = td["current_node"].squeeze(-1)

        locs = td["locs"]  # [B, N+1, 2]
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        td["tour_length"] = td["tour_length"] + distance
        td["visited"] = td["visited"].scatter(1, action.unsqueeze(-1), True)
        td["current_node"] = action.unsqueeze(-1)
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        # Capacity update
        at_depot = action == 0
        n_cust = td["demand"].shape[-1]
        cust_idx = (action - 1).clamp(min=0, max=n_cust - 1)
        demand_at_node = td["demand"].gather(-1, cust_idx.unsqueeze(-1)).squeeze(-1)
        demand_at_node = demand_at_node * (~at_depot).float()

        new_used = td["used_capacity"] + demand_at_node
        # Reset used capacity when returning to depot
        td["used_capacity"] = torch.where(at_depot, torch.zeros_like(new_used), new_used)

        return td

    # ------------------------------------------------------------------
    # Done / Mask / Reward
    # ------------------------------------------------------------------

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Done when all customers are visited and vehicle is at depot.

        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        all_customers_done = td["visited"][..., 1:].all(dim=-1)
        at_depot = td["current_node"].squeeze(-1) == 0
        step = td["i"].squeeze(-1) if td["i"].dim() > 1 else td["i"]
        return all_customers_done & at_depot & (step > 0)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Action mask for CVRP.

        A customer is valid if:
          (a) not yet visited, AND
          (b) its demand does not exceed remaining vehicle capacity.

        The depot (index 0) is valid UNLESS the vehicle is already at the
        depot while unvisited customers remain.
        Args:
            td: Input TensorDict containing the environment state.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        vehicle_cap = td["vehicle_capacity"]  # [B]
        if vehicle_cap.dim() > 1:
            vehicle_cap = vehicle_cap.squeeze(-1)

        remaining = (vehicle_cap - td["used_capacity"]).unsqueeze(-1)  # [B, 1]
        demand = td["demand"]  # [B, N]

        exceeds_cap = demand > remaining + 1e-5  # [B, N]
        visited_cust = td["visited"][..., 1:]  # [B, N]
        mask_cust = ~(visited_cust | exceeds_cap)  # [B, N]: True = valid customer

        # Depot: invalid if already at depot and unserved customers exist
        at_depot = td["current_node"].squeeze(-1) == 0  # [B]
        has_valid_cust = mask_cust.any(dim=-1)  # [B]
        mask_depot = ~(at_depot & has_valid_cust)  # [B]: True = depot valid

        return torch.cat([mask_depot.unsqueeze(-1), mask_cust], dim=-1)  # [B, N+1]

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reward is the negative total tour length.

        Args:
            td: Input TensorDict containing the environment state.
            actions: Tensor of actions to take.

        Returns:
            Updated TensorDict or tensor containing the result.
        """
        return -td["tour_length"]
