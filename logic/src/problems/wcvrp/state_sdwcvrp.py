"""
State representation for the Split Delivery Waste Collection Vehicle Routing Problem (SDWCVRP).

This module defines the StateSDWCVRP class, which allows for bins to be
collected over multiple visits if the vehicle capacity is insufficient.
"""

import torch
import torch.nn.functional as F

from typing import NamedTuple
from logic.src.utils.definitions import VEHICLE_CAPACITY


class StateSDWCVRP(NamedTuple):
    """
    Data class representing the state of an SDWCVRP tour.

    In SDWCVRP, a single node can be visited multiple times to collect parts
     of its waste. This state tracks the 'wastes_with_depot' dynamically
    deceasing as waste is collected.
    """

    # Fixed input
    coords: torch.Tensor
    waste: torch.Tensor

    # Cost function weights
    w_waste: float
    w_length: float
    w_overflows: float

    # Maximum amount of waste before bin is considered to be overflowing
    max_waste: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and wastes tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    wastes_with_depot: torch.Tensor  # Keeps track of remaining wastes
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    cur_overflows: torch.Tensor
    cur_total_waste: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    edges: torch.Tensor
    dist_matrix: torch.Tensor
    vehicle_capacity: float

    def __getitem__(self, key):
        """Allows indexing the state for batch operations."""
        if torch.is_tensor(key) or isinstance(
            key, slice
        ):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                wastes_with_depot=self.wastes_with_depot[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                cur_overflows=self.cur_overflows[key],
                cur_total_waste=self.cur_total_waste[key],
            )
        return self[key]

    @staticmethod
    def initialize(input, edges, cost_weights=None, dist_matrix=None, hrl_mask=None):
        """
        Initializes the state for a batch of SDWCVRP instances.

        Args:
            input (dict): Input data containing 'depot', 'loc', 'waste', 'max_waste'.
            edges (Tensor): Edge indices.
            cost_weights (dict, optional): Weights for overflows, length, and waste.
            dist_matrix (Tensor, optional): Precomputed distance matrix.
            hrl_mask (Tensor, optional): Mask for Hierarchical RL.

        Returns:
            StateSDWCVRP: The initialized state.
        """
        depot = input["depot"]
        loc = input["loc"]
        waste = input["waste"]
        max_waste = input["max_waste"]
        batch_size, n_loc, _ = loc.size()

        # Pad waste with a 0 at index 0 for the depot to prevent indexing errors
        waste = F.pad(waste, (1, 0), mode="constant", value=0)

        res = StateSDWCVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            waste=waste,
            max_waste=max_waste[:, None],
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[
                :, None
            ],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=waste.new_zeros(batch_size, 1),
            wastes_with_depot=waste[:, None, :],
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input["depot"][:, None, :],  # Add step dimension
            cur_overflows=torch.sum((input["waste"] >= max_waste[:, None]), dim=-1),
            cur_total_waste=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(
                1, dtype=torch.int64, device=loc.device
            ),  # Vector with length num_steps
            w_waste=1 if cost_weights is None else cost_weights["waste"],
            w_overflows=1 if cost_weights is None else cost_weights["overflows"],
            w_length=1 if cost_weights is None else cost_weights["length"],
            edges=edges,
            dist_matrix=dist_matrix,
            vehicle_capacity=VEHICLE_CAPACITY,
        )

        if hrl_mask is not None:
            # hrl_mask is (Batch, N).
            # wastes_with_depot is (Batch, 1, N+1). Index 0 is Depot.
            # Mask out nodes that shouldn't be visited by setting waste to 0
            if hrl_mask.dim() == 2:
                if hrl_mask.size(1) != n_loc:
                    print(
                        f"CRITICAL ERROR: HRL Mask width {hrl_mask.size(1)} != n_loc {n_loc}"
                    )
                res.wastes_with_depot[:, 0, 1:] = res.wastes_with_depot[:, 0, 1:] * (
                    1 - hrl_mask.to(waste.dtype)
                )
            else:
                res.wastes_with_depot[:, 0, 1:] = res.wastes_with_depot[:, 0, 1:] * (
                    1 - hrl_mask.squeeze().to(waste.dtype)
                )

        return res

    def get_final_cost(self):
        """Returns the final objective value, including return to depot."""
        assert self.all_finished()
        length_cost = self.w_length * self.lengths + self.w_length * (
            self.coords[self.ids, 0, :] - self.cur_coord
        ).norm(p=2, dim=-1)
        return (
            self.w_overflows * self.cur_overflows
            + length_cost
            - self.w_waste * self.cur_total_waste
        )

    def update(self, selected):
        """
        Updates the state after selecting the next node, potentially collecting
        partial waste.

        Args:
            selected (Tensor): Index of the selected node.

        Returns:
            StateSDWCVRP: The updated state.
        """
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(
            p=2, dim=-1
        )  # (batch_dim, 1)

        # Not selected_waste is waste of first node (by clamp) so incorrect for nodes that visit depot!
        selected_waste = self.wastes_with_depot.gather(-1, prev_a[:, :, None])[
            :, :, 0
        ].clamp(max=self.max_waste[self.ids, 0])
        delivered_waste = torch.min(
            selected_waste, self.vehicle_capacity - self.used_capacity
        )
        cur_total_waste = self.cur_total_waste + selected_waste

        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (self.used_capacity + delivered_waste) * (prev_a != 0).float()

        # Update wastes_with_depot by subtracting amount delivered
        wastes_with_depot = self.wastes_with_depot.scatter(
            -1,
            prev_a[:, :, None],
            self.wastes_with_depot.gather(-1, prev_a[:, :, None])
            - delivered_waste[:, :, None],
        )

        # Update the number of overflows
        cur_overflows = self.cur_overflows - torch.sum(
            (self.waste[self.ids, selected] >= self.max_waste) & (selected != 0), dim=-1
        )

        return self._replace(
            prev_a=prev_a,
            used_capacity=used_capacity,
            wastes_with_depot=wastes_with_depot,
            lengths=lengths,
            cur_coord=cur_coord,
            cur_overflows=cur_overflows,
            cur_total_waste=cur_total_waste,
            i=self.i + 1,
        )

    def all_finished(self):
        """Checks if all instances have returned to the depot after starting."""
        return (self.i > 0).all() and (self.prev_a == 0).all()

    def get_current_node(self):
        """Returns the current node index."""
        return self.prev_a

    def get_remaining_overflows(self):
        """Returns the number of remaining overflows."""
        return self.cur_overflows.unsqueeze(-1)

    def get_current_efficiency(self):
        """Returns the current collection efficiency."""
        efficiency = self.cur_total_waste / self.lengths
        return torch.nan_to_num(efficiency, nan=0.0)

    def get_mask(self):
        """
        Gets a mask of feasible next actions.

        Nodes with no remaining waste or when the vehicle is at full capacity
        are masked. Depot is allowed unless there are valid customers to explore
        and we are currently at the depot.

        Returns:
            Tensor: A boolean mask where True indicates an infeasible action.
        """
        # Nodes that cannot be visited are already visited or too much waste to be served now
        mask_loc = (self.wastes_with_depot[:, :, 1:] == 0) | (
            self.used_capacity[:, :, None] >= self.vehicle_capacity
        )

        # For partial tours, we always allow the depot as a valid next action to finish
        mask_depot = torch.zeros_like(self.prev_a, dtype=torch.bool)

        # Prevent 0 -> 0 transition (staying at depot) to force exploration
        has_valid_customer = ~mask_loc.all(dim=-1)
        mask_depot = mask_depot | ((self.prev_a == 0) & has_valid_customer)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1).bool()

    def get_edges_mask(self):
        """Returns the edge mask for the current node."""
        batch_size, n_coords, _ = self.coords.size()
        if self.i.item() == 0:
            return torch.zeros(
                batch_size, 1, n_coords, dtype=torch.uint8, device=self.coords.device
            )
        else:
            return self.edges.gather(
                1, self.prev_a.unsqueeze(-1).expand(-1, -1, n_coords)
            )

    def get_edges(self):
        """Returns graph edges."""
        return self.edges

    def construct_solutions(self, actions):
        """Returns actions sequence as solutions."""
        return actions
