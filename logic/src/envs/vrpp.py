"""
VRPP Environment implementation.

Vehicle Routing Problem with Profits: Select profitable subset
of nodes to visit while minimizing travel cost.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.envs.generators import VRPPGenerator


class VRPPEnv(RL4COEnvBase):
    """
    Vehicle Routing Problem with Profits Environment.

    The agent must select which nodes to visit to maximize
    total prize collected minus travel cost.
    """

    NAME = "vrpp"
    name: str = "vrpp"

    def __init__(
        self,
        generator: Optional[VRPPGenerator] = None,
        generator_params: Optional[dict] = None,
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """
        Initialize VRPPEnv.

        Args:
            generator: Problem instance generator.
            generator_params: Parameters for generator initialization.
            waste_weight: Weight for waste collection in reward.
            cost_weight: Weight for travel cost in reward.
            revenue_kg: Optional revenue per kg (overrides waste_weight).
            cost_km: Optional cost per km (overrides cost_weight).
            device: Device for torch tensors ('cpu' or 'cuda').
            **kwargs: Additional keyword arguments.
        """
        generator_params = generator_params or kwargs
        if generator is None:
            generator = VRPPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)
        self.waste_weight = revenue_kg if revenue_kg is not None else waste_weight
        self.cost_weight = cost_km if cost_km is not None else cost_weight

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """
        Initialize VRPP episode state with depot prepending and state field initialization.

        This method sets up the initial state for a Vehicle Routing Problem with Profits (VRPP)
        episode. It handles depot prepending logic to ensure coordinates are in the correct format
        (N+1 nodes including depot at index 0), and initializes all tracking fields.

        State Initialization Logic:
            1. Check if depot needs to be prepended to customer locations
            2. Prepend depot to both locations and waste arrays if needed
            3. Initialize tracking fields: visited, tour, tour_length, collected_waste

        Args:
            td (TensorDict): Input tensor dictionary containing:
                - locs (Tensor): Location coordinates, shape (batch, N, 2) or (batch, N+1, 2)
                - depot (Tensor): Depot coordinates, shape (batch, 2)
                - waste (Tensor, optional): Waste/prize at each location, shape (batch, N) or (batch, N+1)
                - visited (Tensor, optional): If present, skip reinitialization (transductive search)

        Returns:
            TensorDict: Updated state dictionary with added fields:
                - current_node (LongTensor): Current position index, shape (batch, 1), initialized to 0 (depot)
                - visited (BoolTensor): Visit status for each node, shape (batch, num_nodes), depot marked visited
                - tour (LongTensor): Sequence of visited node indices, shape (batch, 0), starts empty
                - tour_length (Tensor): Total distance traveled, shape (batch,), initialized to 0.0
                - collected_waste (Tensor): Total waste/profit collected, shape (batch,), initialized to 0.0

        Note:
            The robust depot prepending logic handles three scenarios:
            1. Generator provides num_loc - compare actual shape to expected N+1
            2. No generator hint - check if first location matches depot coordinates
            3. Mismatched waste array - ensure waste has same node count as locations
        """
        # If state is already initialized (e.g. transductive search resuming), skip coord mod
        if "visited" in td.keys():
            return td

        device = td.device
        bs = td.batch_size

        # Prepend depot to locs and waste if they are customer-only
        locs = td["locs"]
        waste = td.get("waste")

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

        # Initialize state fields
        td["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        td["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        td["visited"][..., 0] = True  # Depot index 0

        # Tour tracking
        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, device=device)
        td["collected_waste"] = torch.zeros(*bs, device=device)
        return td

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Execute a routing action and update the episode state.

        This method processes a single step in the VRPP episode by:
        1. Computing the Euclidean distance traveled from current node to selected action node
        2. Collecting waste/profit from the destination node (if unvisited and not depot)
        3. Updating the visit mask, current position, tour sequence, and cumulative metrics

        Reward Logic:
            - Waste is collected only from nodes that are:
                a) Previously unvisited (checked via visited mask)
                b) Not the depot (action != 0)
            - The depot (node 0) has zero waste and serves as the return point
            - Revisiting a node does not collect additional waste

        Args:
            td (TensorDict): Current state containing:
                - action (LongTensor): Selected next node index, shape (batch,)
                - current_node (LongTensor): Current position, shape (batch, 1)
                - locs (Tensor): All node coordinates including depot, shape (batch, num_nodes, 2)
                - waste (Tensor): Waste/profit at each node, shape (batch, num_nodes)
                - visited (BoolTensor): Visit status, shape (batch, num_nodes)
                - tour (LongTensor): Historical tour, shape (batch, tour_len)
                - tour_length (Tensor): Cumulative distance, shape (batch,)
                - collected_waste (Tensor): Cumulative waste collected, shape (batch,)

        Returns:
            TensorDict: Updated state with modifications to:
                - tour_length: Incremented by Euclidean distance
                - collected_waste: Incremented by waste[action] if valid collection
                - visited: action node marked as True
                - current_node: Updated to action value
                - tour: Action appended to tour sequence

        Note:
            The method handles edge cases where waste array may not include depot
            (shape mismatch) by dynamically prepending a zero depot waste value.
        """
        # Locs and waste are already prepended in reset or previously
        bs = td.batch_size
        device = td.device
        locs = td["locs"]
        waste = td["waste"]

        action = td["action"]
        current = td["current_node"].squeeze(-1)

        # Compute distance traveled
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        # Update tour length
        td["tour_length"] = td["tour_length"] + distance

        # Collect prize (only for unvisited, non-depot nodes)
        is_new_visit = ~td["visited"].gather(1, action.unsqueeze(-1)).squeeze(-1)
        is_not_depot = action != 0

        # waste handling
        waste = td.get("waste")
        if waste is not None and waste.shape[-1] == td["visited"].shape[-1] - 1:
            # needs depot
            waste = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)
            td["waste"] = waste

        if waste is not None:
            waste_collected = waste.gather(1, action.unsqueeze(-1)).squeeze(-1)
            td["collected_waste"] = (
                td["collected_waste"] + waste_collected * is_new_visit.float() * is_not_depot.float()
            )

        # Update visited
        td["visited"] = td["visited"].scatter(1, action.unsqueeze(-1), True)

        # Update current node
        td["current_node"] = action.unsqueeze(-1)

        # Append to tour
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute action mask for VRPP with must-go constraints.

        Standard behavior:
        - Can visit any unvisited node
        - Can return to depot at any time
        - Cannot visit already visited nodes (except depot)

        Must-go behavior:
        - If must_go is None: Standard behavior (depot always valid)
        - If must_go has True values: Must route to those bins; depot invalid until done
        - If must_go is all False: No routing needed; depot is valid (stay)

        Returns:
            Tensor: Boolean mask (batch, num_nodes) where True = valid action.
        """
        mask = ~td["visited"].clone()

        # Must-go routing logic
        must_go = td.get("must_go", None)

        if must_go is not None:
            # must_go: (batch, num_nodes) boolean tensor
            # True = must visit this bin, False = optional

            # Pending must-go bins: must_go AND not yet visited
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
        Compute VRPP reward: waste - cost.

        Reward = (waste_weight * total_waste) - (cost_weight * tour_length)
        """
        waste = td.get("collected_waste", torch.zeros_like(td["tour_length"]))
        cost = td["tour_length"]

        # Also add return to depot distance if not already there
        current = td["current_node"].squeeze(-1)
        # Combine depot and customers for full locs if not already done
        locs = td["locs"]
        if locs.shape[-2] == td.get("visited", torch.zeros(0)).shape[-1] - 1:
            locs = torch.cat([td["depot"].unsqueeze(1), locs], dim=1)

        current_idx = current[:, None, None].expand(-1, -1, 2)
        current_loc = locs.gather(1, current_idx).squeeze(1)
        depot_loc = td["depot"]
        return_distance = torch.norm(depot_loc - current_loc, dim=-1)

        # Only add return distance if not already at depot
        not_at_depot = current != 0
        total_cost = cost + return_distance * not_at_depot.float()

        # Store decomposed rewards for GDPO
        td["reward_waste"] = waste
        td["reward_cost"] = -total_cost  # Convention: maximized, so negative cost

        reward = self.waste_weight * waste - self.cost_weight * total_cost
        return reward

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Episode is done when agent returns to depot."""
        current = td["current_node"].squeeze(-1)
        step = td["i"].squeeze(-1)
        return (current == 0) & (step > 0)
