"""
Pickup and Delivery Problem (PDP) Environment.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.envs.generators import PDPGenerator


class PDPEnv(RL4COEnvBase):
    """
    Pickup and Delivery Problem (PDP) Environment.

    The agent must visit pickup and delivery pairs.
    Precedence Constraint: Pickup 'i' must be visited before Delivery 'i'.
    Capacity Constraint: Vehicle has max capacity. Pickups add load, Deliveries remove it.

    Convention:
    - Nodes 0 to N-1 are Pickups.
    - Nodes N to 2N-1 are Deliveries.
    - Node 'i' corresponds to Delivery 'i + N'.
    - Depot is maintained separately in 'depot' but logically is usually node 0 in attention models.
      Indices in action/visited usually map 1..2N to customers, 0 to depot.
    """

    name: str = "pdp"

    def __init__(
        self,
        generator: Optional[PDPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """Initialize PDPEnv."""
        generator_params = generator_params or kwargs
        if generator is None:
            generator = PDPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize PDP state."""
        device = td.device
        bs = td.batch_size

        # locs shape: (batch, 2*num_loc, 2)
        # num_loc here is actually the number of pairs N.
        # So total nodes in locs is 2N.
        num_pairs = td["locs"].shape[-2] // 2
        num_nodes = 2 * num_pairs + 1  # +1 for depot

        # Current node (start at depot at index 0)
        td["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["visited"].index_fill_(-1, torch.tensor([0], device=device), True)

        # Current load: start empty
        td["current_load"] = torch.zeros(*bs, 1, dtype=torch.float, device=device)

        # Tour tracking
        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, device=device)

        return td

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Execute action and update state."""
        action = td["action"]  # [batch]
        action_idx = action[:, None]  # [batch, 1]

        # Combine depot and customers for coordinates
        depot = td["depot"]
        customers = td["locs"]
        # Standard RL4CO convention: node 0 is depot, 1..2N are customers
        # (batch, 2*N+1, 2)
        locs = torch.cat([depot.unsqueeze(1), customers], dim=1)

        # 1. Update distance
        current_node_idx = td["current_node"]
        current_pos = locs.gather(1, current_node_idx[:, :, None].expand(-1, -1, 2)).squeeze(1)
        next_pos = locs.gather(1, action_idx[:, :, None].expand(-1, -1, 2)).squeeze(1)
        dist = torch.norm(next_pos - current_pos, dim=-1)
        td["tour_length"] = td["tour_length"] + dist

        # 2. Update visited
        td["visited"] = td["visited"].scatter(1, action_idx, True)

        # 3. Update load
        # Pickups (indices 1..N) increase load +1
        # Deliveries (indices N+1..2N) decrease load -1
        # Depot (0) doesn't change load

        num_pairs = customers.shape[1] // 2

        # Mask for pickups: 1 <= idx <= N
        is_pickup = (action >= 1) & (action <= num_pairs)
        # Mask for deliveries: N+1 <= idx <= 2N
        is_delivery = action > num_pairs

        load_change = torch.zeros_like(td["current_load"]).squeeze(-1)
        load_change[is_pickup] = 1.0
        load_change[is_delivery] = -1.0

        td["current_load"] = td["current_load"] + load_change.unsqueeze(-1)

        # 4. Update current node
        td["current_node"] = action_idx

        # 5. Append to tour
        td["tour"] = torch.cat([td["tour"], action_idx], dim=-1)

        return td

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute valid actions.
        Constraints:
        1. Cannot visit visited nodes.
        2. Pickup 'i' valid if not visited and load < capacity. (Assuming load=1 per item, capacity=inf? No, usually capacity=1 or fixed)
           Let's assume standard PDP with unit demands and fixed vehicle capacity if provided, or uncapacitated if not.
           Ideally `capacity` should be in td. If not, assume infinite or 1?
           Usually simple PDP is uncapacitated vehicle-wise OR capacity=1 (stack).
           However, PDPGenerator doesn't put capacity in td (my bad, I missed it).
           RL4CO's PDP typically uses uncapacitated vehicle with respect to weight, or unit capacity?
           Wait, generator output: locs, depot. No capacity.
           Let's assume uncapacitated for now or make capacity a parameter.
           BUT the constraint "Pickup before Delivery" is essentially a capacity of sorts on the 'pair'.

        Let's implement the core precedence constraint:
        - Delivery 'i+N' is valid ONLY IF Pickup 'i' is visited.

        Refining constraints:
        - Node is unvisited.
        - If Node is Pickup (1..N): Valid if load < capacity (if capacity exists).
        - If Node is Delivery (N+1..2N): Valid if Pickup (node - N) is visited.
        - Depot (0): Valid if all Deliveries visited (which implies all pickups visited).

        Refinement on Capacity: I should probably add capacity to generator or allow it in Env.
        For now, let's assume standard "Capacity=1" vehicle if not specified (pick one, deliver one?),
        OR Infinite capacity (just routing).
        The standard Li & Lim benchmarks have capacity.
        Usually `capacity` tensor is in td.
        I will modify code to get capacity from td or default to infinity.
        """
        num_nodes = td["visited"].shape[-1]
        num_pairs = (num_nodes - 1) // 2

        # 1. Basic visited mask (False if visited)
        mask = ~td["visited"]

        # 2. Depot (0) is only valid if all requests are done
        # We check if all deliveries (indices N+1 .. 2N) are visited
        # Deliveries indices: num_pairs+1 to 2*num_pairs
        delivery_indices = torch.arange(num_pairs + 1, 2 * num_pairs + 1, device=td.device)
        deliveries_visited = td["visited"][..., delivery_indices].all(dim=-1)

        # Depot mask logic:
        # If not all done, depot is invalid (unless we allow intermediate depot visits? usually no)
        # Note: visited[0] is always True, so ~visited[0] is False.
        # We need to explicitly set 0 to True if all done.

        # Default: depot is invalid (already visited)
        # We want to allow closing the tour.
        # But wait, standard masking usually masks 'already visited'.
        # For depot, we usually allow re-visiting ONLY at the end.
        mask[..., 0] = deliveries_visited

        # 3. Delivery Precedence
        # Delivery j+N valid only if Pickup j visited
        # Pickup indices: 1 to N
        # Delivery indices: N+1 to 2N
        pickup_visited = td["visited"][..., 1 : num_pairs + 1]

        # This expands to shape (batch, num_pairs)
        # We need to apply this to the corresponding delivery columns in mask
        # mask[..., N+1 : 2N+1] &= pickup_visited
        mask[..., num_pairs + 1 :] = mask[..., num_pairs + 1 :] & pickup_visited

        # 4. Capacity (if present)
        # If we have capacity constraint:
        # Pickups valid only if current_load < capacity
        if "capacity" in td.keys():
            capacity = td["capacity"]  # [batch] or [batch, 1]
            current_load = td["current_load"]  # [batch, 1]

            # If load >= capacity, all pickups are invalid
            # Pickups are 1..N
            can_pickup = (current_load < capacity).expand(-1, num_pairs)  # (batch, num_pairs)
            mask[..., 1 : num_pairs + 1] &= can_pickup

        return mask

    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reward is negative total tour length."""
        return -td["tour_length"]

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Done when all deliveries visited and back at depot."""
        # Deliveries are indices N+1 .. 2N
        # But simply checking 'visited' all is enough because:
        # - Depot (0) visited at start.
        # - Pickups visited to enable Deliveries.
        # - Deliveries visited.
        # - AND current node is depot (0).

        all_visited = td["visited"].all(dim=-1)
        is_depot = td["current_node"].squeeze(-1) == 0
        return all_visited & is_depot
