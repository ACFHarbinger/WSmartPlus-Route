"""
WCVRP Environment implementation.

Waste Collection Vehicle Routing Problem: Collect waste from bins
while respecting vehicle capacity constraints.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.envs.generators import WCVRPGenerator


class WCVRPEnv(RL4COEnvBase):
    """
    Waste Collection VRP Environment.

    The agent must collect waste from bins before they overflow,
    while minimizing travel cost and respecting vehicle capacity.
    """

    NAME = "wcvrp"
    name: str = "wcvrp"

    def __init__(
        self,
        generator: Optional[WCVRPGenerator] = None,
        generator_params: Optional[dict] = None,
        overflow_penalty: float = 10.0,
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """
        Initialize WCVRPEnv.

        Args:
            generator: Problem instance generator.
            generator_params: Parameters for generator initialization.
            overflow_penalty: Penalty for bin overflow.
            waste_weight: Reward weight for waste collection.
            cost_weight: Weight for travel cost in reward.
            revenue_kg: Optional revenue per kg (overrides waste_weight).
            cost_km: Optional cost per kg (overrides cost_weight).
            device: Device for torch tensors ('cpu' or 'cuda').
            **kwargs: Additional keyword arguments.
        """
        generator_params = generator_params or kwargs
        if generator is None:
            generator = WCVRPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)
        self.overflow_penalty = overflow_penalty
        self.waste_weight = revenue_kg if revenue_kg is not None else waste_weight
        self.cost_weight = cost_km if cost_km is not None else cost_weight

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """
        Initialize Waste Collection VRP (WCVRP) episode state with capacity tracking.

        This method extends the base VRPP reset logic to add capacity constraints and overflow
        monitoring for waste collection scenarios. It initializes bin fill levels, vehicle load
        tracking, and overflow counting.

        Capacity-Specific Initialization:
            - current_load: Vehicle's current waste load (starts at 0)
            - total_collected: Cumulative waste collected across all depot returns
            - cur_overflows: Count of bins currently at or exceeding max_waste threshold

        Args:
            td (TensorDict): Input tensor dictionary containing:
                - locs (Tensor): Location coordinates, shape (batch, N, 2) or (batch, N+1, 2)
                - depot (Tensor): Depot coordinates, shape (batch, 2)
                - waste (Tensor): Current bin fill levels, shape (batch, N) or (batch, N+1)
                - max_waste (Tensor): Maximum bin capacity per node, shape (batch, 1) or (batch, N)
                - capacity (Tensor): Vehicle capacity, shape (batch,)
                - visited (Tensor, optional): If present, skip reinitialization

        Returns:
            TensorDict: Updated state with WCVRP-specific fields:
                - current_node (LongTensor): Starting at depot (0), shape (batch,)
                - visited (BoolTensor): Depot marked as visited, shape (batch, num_nodes)
                - tour (LongTensor): Empty tour sequence, shape (batch, 0)
                - tour_length (Tensor): Zero distance, shape (batch,)
                - current_load (Tensor): Zero initial load, shape (batch,)
                - collected_waste (Tensor): Zero collected, shape (batch,)
                - total_collected (Tensor): Zero total, shape (batch,)
                - cur_overflows (Tensor): Count of overflowing bins (waste >= max_waste), shape (batch,)

        Note:
            Overflow counting excludes the depot (index 0) and only considers customer bins.
            The initial overflow count is computed as sum(waste[1:] >= max_waste[1:]).
        """
        # If state already initialized (e.g. transductive search resuming), skip coord mod
        if "visited" in td.keys():
            return td

        device = td.device
        bs = td.batch_size

        # Prepend depot to locs and waste if they are customer-only
        locs = td["locs"]
        waste = td["waste"]

        # Robust N vs N+1 logic
        # gen_n represents customers only. If shape is gen_n + 1, it already has depot.
        gen_n = getattr(self.generator, "num_loc", None)

        needs_prepend = False
        if "depot" in td.keys():
            # 1. If we know N, check if we are already at N+1
            if gen_n is not None and locs.shape[-2] == gen_n + 1:
                needs_prepend = False
            # 2. If we are at N, we definitely need to prepend
            elif gen_n is not None and locs.shape[-2] == gen_n:
                needs_prepend = True
            # 3. Fallback: check coordinate values
            elif not torch.allclose(locs[..., 0, :], td["depot"], atol=1e-4):
                needs_prepend = True

        if needs_prepend:
            td["locs"] = torch.cat([td["depot"].unsqueeze(1), locs], dim=1)
            locs = td["locs"]
            if waste is not None:
                td["waste"] = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)
        elif waste is not None and waste.shape[-1] == locs.shape[-2] - 1:
            # Catch case where locs has depot but waste doesn't
            td["waste"] = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)

        num_nodes = td["locs"].shape[-2]

        td["current_node"] = torch.zeros(bs, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["visited"][..., 0] = True

        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, device=device)
        td["current_load"] = torch.zeros(*bs, device=device)
        td["collected_waste"] = torch.zeros(*bs, device=device)
        td["total_collected"] = torch.zeros(*bs, device=device)

        # Calculate initial overflows (consistent with _get_reward: count of nodes >= max_waste)
        max_waste = td.get("max_waste", torch.tensor(1.0, device=device))
        if max_waste.dim() == 1:
            mw = max_waste.unsqueeze(-1)
        else:
            mw = max_waste

        # Customers only (skip depot at index 0)
        waste_cust = td["waste"][..., 1:]
        max_w_cust = mw if mw.shape[-1] == 1 else mw[..., 1:]

        td["cur_overflows"] = (waste_cust >= max_w_cust).float().sum(-1)

        return td

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Execute a waste collection action with capacity constraints and bin clearing.

        This method extends the base VRPP step logic to handle:
        1. Capacity-constrained waste collection (collect min of bin level and vehicle capacity)
        2. Depot emptying (vehicle load resets to 0 when returning to depot)
        3. Bin clearing (collected bins have waste set to 0)
        4. Load tracking across multiple depot visits

        Collection Mechanics:
            - At customer nodes (action != 0):
                * Collect min(waste_at_node, max_waste, remaining_capacity)
                * Add collected amount to current_load and total_collected
                * Set bin waste level to 0 after collection
            - At depot (action == 0):
                * Reset current_load to 0 (emptying vehicle)
                * total_collected remains unchanged (cumulative across trips)

        Args:
            td (TensorDict): Current state containing:
                - action (LongTensor): Selected node to visit, shape (batch,)
                - current_node (LongTensor): Current position, shape (batch,)
                - current_load (Tensor): Current vehicle load, shape (batch,)
                - waste (Tensor): Bin fill levels, shape (batch, num_nodes)
                - max_waste (Tensor): Bin capacity limits, shape (batch, 1) or (batch, num_nodes)
                - capacity (Tensor): Vehicle capacity, shape (batch,)
                - visited, tour, tour_length: Inherited tracking fields

        Returns:
            TensorDict: Updated state with modifications to:
                - current_load: Increased by collected waste, or reset to 0 at depot
                - collected_waste: Incremented by collection amount
                - total_collected: Cumulative waste across all trips (equals collected_waste)
                - waste: Bin at action index set to 0 after collection
                - current_node: Updated to action value
                - tour, tour_length, visited: Updated via parent class

        Note:
            The method calls super()._step_instance() first to handle base VRP state updates
            (distance, tour, visited), then applies capacity-specific waste collection logic.
        """
        action = td["action"]
        waste = td["waste"]

        # Core mechanics
        td = super()._step_instance(td)

        is_not_depot = action != 0
        waste = td["waste"]  # Re-fetch updated waste (OpsMixin might have touched it, but we already prepended)
        waste_at_node = waste.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Collect waste (clamped to max_waste if present)
        max_w = td.get("max_waste", torch.tensor(1e9, device=td.device))
        if max_w.dim() > 1 and max_w.shape[-1] == td["visited"].shape[-1] - 1:
            max_w = torch.cat([torch.tensor(1e9, device=td.device).expand(*td.batch_size, 1), max_w], dim=1)
        if max_w.dim() > 1:
            max_w_at_node = max_w.gather(1, action.unsqueeze(-1)).squeeze(-1)
        else:
            max_w_at_node = max_w

        collected = torch.min(waste_at_node, max_w_at_node) * is_not_depot.float()
        td["current_load"] = td["current_load"] + collected
        td["collected_waste"] = td["collected_waste"] + collected
        td["total_collected"] = td["collected_waste"]  # Alias

        # Empty at depot
        at_depot = action == 0
        td["current_load"] = torch.where(at_depot, torch.zeros_like(td["current_load"]), td["current_load"])

        # Clear bin (set waste to 0 after collection)
        td["waste"].scatter_(1, action.unsqueeze(-1), 0)

        # Update current node (overriding super which might used unsqueezed action)
        td["current_node"] = action.squeeze(-1) if action.dim() > 1 else action

        return td

    def _step(self, td: TensorDict) -> TensorDict:
        """step.

        Args:
            td (TensorDict): Description of td.

        Returns:
            Any: Description of return value.
        """
        return super(WCVRPEnv, self)._step(td)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute action mask based on capacity, visited status, and must-go constraints.

        This method implements three-tier filtering to determine valid actions:
        1. **Visit filter**: Exclude already-visited nodes (except depot)
        2. **Capacity filter**: Exclude bins exceeding remaining vehicle capacity
        3. **Must-go filter**: Enforce priority routing to critical bins

        Must-Go Routing Logic:
            - If must_go is None: Standard VRP behavior, depot always valid
            - If must_go has True values: MUST visit those bins first; depot blocked until complete
            - If must_go is all False: Episode complete; only depot is valid (return home)

        Edge Cases:
            - **Vehicle full with must-go bins**: Depot becomes valid to allow emptying
            - **All must-go bins visited**: Depot becomes valid for return
            - **No valid customer nodes**: Depot is always fallback to prevent deadlock
            - **Waste array mismatch**: Dynamically prepends depot zero if needed

        Args:
            td (TensorDict): Current state containing:
                - visited (BoolTensor): Visit status, shape (batch, num_nodes)
                - waste (Tensor): Bin fill levels, shape (batch, num_nodes) or (batch, N)
                - current_load (Tensor): Current vehicle load, shape (batch,)
                - capacity (Tensor): Vehicle capacity, shape (batch,)
                - must_go (BoolTensor, optional): Priority bins, shape (batch, num_nodes)

        Returns:
            Tensor: Boolean action mask (batch, num_nodes) where True indicates a valid action.
                    At least one action per batch instance is guaranteed to be valid.

        Note:
            The capacity check uses remaining_capacity = capacity - current_load.
            Nodes requiring more waste collection than remaining capacity are masked out,
            forcing a depot return when the vehicle is near full.
        """
        mask = ~td["visited"].clone()

        # Check capacity constraints
        waste = td["waste"]
        if waste.shape[-1] == mask.shape[-1] - 1:
            waste = torch.cat([torch.zeros(*td.batch_size, 1, device=td.device), waste], dim=1)

        remaining_capacity = td["capacity"].unsqueeze(-1) - td["current_load"].unsqueeze(-1)
        exceeds_capacity = waste > remaining_capacity

        mask = mask & ~exceeds_capacity

        # Must-go routing logic
        must_go = td.get("must_go", None)

        if must_go is not None:
            # must_go: (batch, num_nodes) boolean tensor
            # True = must visit this bin, False = optional

            # Pending must-go bins: must_go AND not yet visited AND within capacity
            pending_must_go = must_go & mask

            # Check if any must-go bins remain (excluding depot at index 0)
            has_pending_must_go = pending_must_go[:, 1:].any(dim=1)

            # Depot is valid only if no pending must-go bins remain
            # If has_pending_must_go is True -> depot invalid (False)
            # If has_pending_must_go is False -> depot valid (True)
            mask[:, 0] = ~has_pending_must_go
        else:
            # No must-go constraint: depot always valid
            mask[:, 0] = True

        return mask

    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute WCVRP reward.

        Efficient routing (low cost, high collection) = high reward.
        """
        collection = td["total_collected"]
        cost = td["tour_length"]

        # Add return to depot
        current = td["current_node"]
        if current.dim() > 1:
            current = current.squeeze(-1)
        if current.dim() == 0:
            current = current.unsqueeze(0)

        locs = td["locs"]
        # Use robust concatenation
        if locs.shape[-2] == td.get("visited", torch.zeros(0)).shape[-1] - 1:
            locs = torch.cat([td["depot"].unsqueeze(1), locs], dim=1)

        # Use robust indexing
        current_idx = current[:, None, None].expand(-1, -1, 2)
        current_loc = locs.gather(1, current_idx).squeeze(1)
        depot_loc = td["depot"]
        return_distance = torch.norm(depot_loc - current_loc, dim=-1)

        not_at_depot = current != 0
        total_cost = cost + return_distance * not_at_depot.float()

        # Calculate overflows: unvisited nodes where waste >= max_waste
        # In current WCVRP, visited nodes have waste=0 in the final td.
        max_waste = td.get("max_waste", torch.tensor(1.0, device=td.device))

        # We only care about customers (index 1 onwards) for overflows
        waste = td["waste"][..., 1:]
        if max_waste.dim() > 1:  # (B, N)
            max_waste = max_waste[..., 1:]
        elif max_waste.dim() == 1:
            max_waste = max_waste.unsqueeze(-1)

        overflows = (waste >= max_waste).float().sum(-1)

        # Store individual components in TensorDict for logging/meta access
        td["collection"] = collection
        td["cost"] = total_cost
        td["overflows"] = overflows

        # Store decomposed rewards for GDPO (signed for maximization)
        td["reward_waste"] = collection
        td["reward_cost"] = -total_cost
        td["reward_overflow"] = -overflows

        reward = self.waste_weight * collection - self.cost_weight * total_cost - self.overflow_penalty * overflows

        # Ensure it's 1D [B] matching batch_size
        if reward.dim() > 1:
            reward = reward.squeeze(-1)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)

        return reward.view(td.batch_size)
