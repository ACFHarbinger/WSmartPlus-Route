"""
State representation for the Capacitated Waste Collection Vehicle Routing Problem (CWCVRP).

This module defines the StateCWCVRP class, which incorporates vehicle capacity
constraints into the waste collection state management.
"""

import torch
import torch.nn.functional as F

from typing import NamedTuple
from logic.src.utils.definitions import VEHICLE_CAPACITY
from logic.src.utils.boolmask import mask_long2bool, mask_long_scatter


class StateCWCVRP(NamedTuple):
    """
    Data class representing the state of a CWCVRP tour.

    In addition to WCVRP tracking, this state manages 'used_capacity' to ensure
    that the vehicle does not exceed its limit between depot visits.
    """

    # Fixed input
    coords: torch.Tensor  # Depot + loc
    waste: torch.Tensor

    # Cost function weights
    w_waste: float
    w_length: float
    w_overflows: float

    # Maximum amount of waste before bin is considered to be overflowing
    max_waste: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and waste tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    cur_overflows: torch.Tensor
    cur_total_waste: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    edges: torch.Tensor
    dist_matrix: torch.Tensor
    vehicle_capacity: float

    @property
    def visited(self):
        """Returns a boolean mask of visited nodes."""
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.waste.size(-1))

    @property
    def dist(self):
        """Calculates the Euclidean distance matrix between all coordinates."""
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(
            p=2, dim=-1
        )

    def __getitem__(self, key):
        """Allows indexing the state for batch operations."""
        if torch.is_tensor(key) or isinstance(
            key, slice
        ):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                cur_overflows=self.cur_overflows[key],
                cur_total_waste=self.cur_total_waste[key],
            )
        return self[key]

    @staticmethod
    def initialize(
        input,
        edges,
        cost_weights=None,
        dist_matrix=None,
        visited_dtype=torch.uint8,
        hrl_mask=None,
    ):
        """
        Initializes the state for a batch of CWCVRP instances.

        Args:
            input (dict): Input data containing 'depot', 'loc', 'waste', 'max_waste'.
            edges (Tensor): Edge indices for the graph model.
            cost_weights (dict, optional): Weights for different cost components.
            dist_matrix (Tensor, optional): Precomputed distance matrix.
            visited_dtype (torch.dtype): Data type for the visited mask.
            hrl_mask (Tensor, optional): Mask for Hierarchical RL.

        Returns:
            StateCWCVRP: The initialized state.
        """
        depot = input["depot"]
        loc = input["loc"]
        waste = input["waste"]
        max_waste = input["max_waste"]
        batch_size, n_loc, _ = loc.size()

        # Initialize visited mask
        if visited_dtype == torch.uint8:
            visited_ = torch.zeros(
                batch_size, 1, n_loc + 1, dtype=torch.uint8, device=loc.device
            )
            if hrl_mask is not None:
                # hrl_mask is (Batch, N).
                # visited_ is (Batch, 1, N+1). Index 0 is Depot.
                # Mark HRL masked nodes as visited (1)
                if hrl_mask.dim() == 2:
                    if hrl_mask.size(1) != n_loc:
                        print(
                            f"CRITICAL ERROR: HRL Mask width {hrl_mask.size(1)} != n_loc {n_loc}"
                        )
                    visited_[:, 0, 1:] = hrl_mask.to(torch.uint8)
                else:
                    # If mask has extra dims
                    visited_[:, 0, 1:] = hrl_mask.squeeze().to(torch.uint8)
        else:
            # Compressed mask logic (int64 bitmask)
            visited_ = torch.zeros(
                batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device
            )
            # NOTE: Compressed mask HRL not implemented yet, assumes uint8 default
            if hrl_mask is not None:
                print("Warning: HRL Mask with compressed visited mask not implemented")

        return StateCWCVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            waste=F.pad(waste, (1, 0), mode="constant", value=0),  # add 0 for depot
            max_waste=max_waste[:, None],
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[
                :, None
            ],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=waste.new_zeros(batch_size, 1),
            visited_=visited_,
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input["depot"][:, None, :],  # Add step dimension
            cur_overflows=torch.sum((input["waste"] >= max_waste[:, None]), dim=-1),
            cur_total_waste=torch.zeros(batch_size, 1, device=loc.device),
            # cur_waste_lost=torch.sum((waste - max_waste[:, None]).clamp(min=0), dim=-1),
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

    def get_final_cost(self):
        """Returns the final objective value, including return to depot."""
        assert self.all_finished()
        length_cost = self.w_length * self.lengths + self.w_length * (
            self.coords[self.ids, 0, :] - self.cur_coord
        ).norm(p=2, dim=-1)
        return (
            self.w_overflows * self.cur_overflows
            + length_cost
            + self.w_waste * self.cur_total_waste
        )

    def update(self, selected):
        """
        Updates the state after selecting the next node.

        Args:
            selected (Tensor): Index of the selected node.

        Returns:
            StateCWCVRP: The updated state.
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

        # selected is (batch, 1) and includes depot at 0
        selected_waste = self.waste.gather(-1, selected).clamp(max=self.max_waste)
        cur_total_waste = self.cur_total_waste + selected_waste

        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (self.used_capacity + selected_waste) * (prev_a != 0).float()

        # Update the number of overflows
        cur_overflows = self.cur_overflows - torch.sum(
            (self.waste[self.ids, selected] >= self.max_waste), dim=-1
        )

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a,
            used_capacity=used_capacity,
            visited_=visited_,
            lengths=lengths,
            cur_coord=cur_coord,
            cur_overflows=cur_overflows,
            cur_total_waste=cur_total_waste,
            i=self.i + 1,
        )

    def all_finished(self):
        """Checks if all instances have returned to the depot after starting."""
        return (self.i > 0).all() and (self.prev_a == 0).all()

    def get_finished(self):
        """Checks if all nodes (including depot) have been visited."""
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        """Returns the current node index."""
        return self.prev_a

    def get_remaining_overflows(self):
        """Returns the number of remaining overflows as a tensor."""
        return self.cur_overflows.unsqueeze(-1)

    def get_current_efficiency(self):
        """Returns the current collection efficiency."""
        efficiency = self.cur_total_waste / self.lengths
        return torch.nan_to_num(efficiency, nan=0.0)

    def get_mask(self):
        """
        Gets a mask of feasible next actions based on capacity and visit status.

        Nodes that would exceed vehicle capacity or have already been visited
        are masked. Depot is allowed unless there are valid customers to explore
        and we are already at the depot.

        Returns:
            Tensor: A boolean mask where True indicates an infeasible action.
        """
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.waste.size(-1))

        # Check which nodes exceed vehicle capacity if visited from current state
        # We clamp waste by bin capacity (max_waste) as that's all we can collect
        waste_to_collect = self.waste[:, 1:].clamp(max=self.max_waste)
        # A node exceeds capacity if truck is not empty AND (current + new > VEHICLE_CAPACITY)
        exceeds_cap = (self.used_capacity[:, :, None] > 0) & (
            waste_to_collect[:, None, :] + self.used_capacity[:, :, None]
            > self.vehicle_capacity
        )

        # Nodes that cannot be visited are already visited or would exceed capacity
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        # Depot can always be visited (to finish the tour) unless restricted below
        mask_depot = torch.zeros_like(self.prev_a, dtype=torch.bool)

        # Prevent 0 -> 0 transition (staying at depot) to force exploration
        # BUT: Only mask depot if there is at least one other valid customer to go to.
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
        """Returns the graph edges."""
        return self.edges

    def construct_solutions(self, actions):
        """Returns the actions sequence as solutions."""
        return actions
