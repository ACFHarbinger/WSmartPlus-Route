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
    total waste collected minus travel cost.
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

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
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
            tensordict (TensorDict): Input tensor dictionary containing:
                - locs (Tensor): Location coordinates, shape (batch, N, 2) or (batch, N+1, 2)
                - depot (Tensor): Depot coordinates, shape (batch, 2)
                - waste (Tensor, optional): Waste at each location, shape (batch, N) or (batch, N+1)
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
        if "visited" in tensordict.keys():
            return tensordict

        device = tensordict.device
        bs = tensordict.batch_size

        # Prepend depot to locs and waste if they are customer-only
        locs = tensordict["locs"]
        waste = tensordict.get("waste")

        # Robust N vs N+1 logic
        # gen_n represents customers only. If shape is gen_n + 1, it already has depot.
        gen_n = getattr(self.generator, "num_loc", None)

        needs_prepend = False
        if "depot" in tensordict.keys():
            # 1. If we know N, check if we are already at N+1
            if gen_n is not None and locs.shape[-2] == gen_n + 1:
                needs_prepend = False
            # 2. If we are at N, we definitely need to prepend
            elif (
                gen_n is not None
                and locs.shape[-2] == gen_n
                or not torch.allclose(locs[..., 0, :], tensordict["depot"], atol=1e-4)
            ):
                needs_prepend = True

        if needs_prepend:
            tensordict["locs"] = torch.cat([tensordict["depot"].unsqueeze(1), locs], dim=1)
            locs = tensordict["locs"]
            if waste is not None:
                tensordict["waste"] = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)
        elif waste is not None and waste.shape[-1] == locs.shape[-2] - 1:
            # Catch case where locs has depot but waste doesn't
            tensordict["waste"] = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)

        num_nodes = tensordict["locs"].shape[-2]

        # Initialize state fields
        tensordict["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        tensordict["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        tensordict["visited"].index_fill_(-1, torch.tensor([0], device=device), True)  # Depot index 0

        # Tour tracking
        tensordict["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        tensordict["tour_length"] = torch.zeros(*bs, device=device)
        tensordict["collected_waste"] = torch.zeros(*bs, device=device)
        return tensordict

    def _step_instance(self, tensordict: TensorDict) -> TensorDict:
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
            tensordict (TensorDict): Current state containing:
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
        bs = tensordict.batch_size
        device = tensordict.device
        locs = tensordict["locs"]
        waste = tensordict["waste"]

        action = tensordict["action"]
        current = tensordict["current_node"].squeeze(-1)

        # Compute distance traveled
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        # Update tour length
        tensordict["tour_length"] = tensordict["tour_length"] + distance

        # Collect waste (only for unvisited, non-depot nodes)
        is_new_visit = ~tensordict["visited"].gather(1, action.unsqueeze(-1)).squeeze(-1)
        is_not_depot = action != 0

        # waste handling
        waste = tensordict.get("waste")
        if waste is not None and waste.shape[-1] == tensordict["visited"].shape[-1] - 1:
            # needs depot
            waste = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)
            tensordict["waste"] = waste

        if waste is not None:
            waste_collected = waste.gather(1, action.unsqueeze(-1)).squeeze(-1)
            tensordict["collected_waste"] = (
                tensordict["collected_waste"] + waste_collected * is_new_visit.float() * is_not_depot.float()
            )

        # Update visited
        tensordict["visited"] = tensordict["visited"].scatter(1, action.unsqueeze(-1), True)

        # Update current node
        tensordict["current_node"] = action.unsqueeze(-1)

        # Append to tour
        tensordict["tour"] = torch.cat([tensordict["tour"], action.unsqueeze(-1)], dim=-1)

        return tensordict

    def _get_action_mask(self, tensordict: TensorDict) -> torch.Tensor:
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

        Args:
            tensordict (TensorDict): Input tensor dictionary containing:
                - visited (BoolTensor): Visit status for each node, shape (batch, num_nodes)
                - must_go (BoolTensor, optional): Must-go constraints, shape (batch, num_nodes)

        Returns:
            Tensor: Boolean mask (batch, num_nodes) where True = valid action.
        """
        mask = ~tensordict["visited"].clone()

        # Must-go routing logic
        must_go = tensordict.get("must_go", None)
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

    def _get_reward(self, tensordict: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute VRPP reward: waste - cost.

        Reward = (waste_weight * total_waste) - (cost_weight * tour_length)

        Args:
            tensordict (TensorDict): Input tensor dictionary containing:
                - collected_waste (Tensor): Total waste collected, shape (batch,)
                - tour_length (Tensor): Total distance traveled, shape (batch,)
                - current_node (LongTensor): Current position index, shape (batch, 1)
                - locs (Tensor): Location coordinates, shape (batch, num_nodes, 2)
                - depot (Tensor): Depot coordinates, shape (batch, 2)
                - visited (BoolTensor): Visit status for each node, shape (batch, num_nodes)

        Returns:
            Tensor: Reward for each batch, shape (batch,)
        """
        waste = tensordict.get("collected_waste", torch.zeros_like(tensordict["tour_length"]))
        cost = tensordict["tour_length"]

        # Also add return to depot distance if not already there
        current = tensordict["current_node"].squeeze(-1)
        # Combine depot and customers for full locs if not already done
        locs = tensordict["locs"]
        if locs.shape[-2] == tensordict.get("visited", torch.zeros(0)).shape[-1] - 1:
            locs = torch.cat([tensordict["depot"].unsqueeze(1), locs], dim=1)

        current_idx = current[:, None, None].expand(-1, -1, 2)
        current_loc = locs.gather(1, current_idx).squeeze(1)
        depot_loc = tensordict["depot"]
        return_distance = torch.norm(depot_loc - current_loc, dim=-1)

        # Only add return distance if not already at depot
        not_at_depot = current != 0
        total_cost = cost + return_distance * not_at_depot.float()

        # Store decomposed rewards for GDPO
        tensordict["reward_waste"] = waste
        tensordict["reward_cost"] = -total_cost  # Convention: maximized, so negative cost

        reward = self.waste_weight * waste - self.cost_weight * total_cost
        return reward

    def _check_done(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Check if the episode is done.

        Args:
            tensordict (TensorDict): Input tensor dictionary containing:
                - current_node (LongTensor): Current position index, shape (batch, 1)
                - i (LongTensor): Current step, shape (batch, 1)

        Returns:
            Tensor: Boolean tensor indicating if the episode is done, shape (batch,)
        """
        current = tensordict["current_node"].squeeze(-1)
        step = tensordict["i"].squeeze(-1)
        return (current == 0) & (step > 0)
