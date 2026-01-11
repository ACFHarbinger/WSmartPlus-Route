"""
State representation for the Stochastic Capacitated Waste Collection Vehicle Routing Problem (SCWCVRP).

This module defines the state of the simulation, including transitions, constraints,
and cost calculations used during the decoding process.
"""

import torch
import torch.nn.functional as F
from typing import NamedTuple
from logic.src.utils.definitions import VEHICLE_CAPACITY
from logic.src.utils.boolmask import mask_long2bool, mask_long_scatter


class StateSCWCVRP(NamedTuple):
    """
    Representation of the current state in an SCWCVRP instance.

    This class maintains both the observed (noisy) and actual (real) state of the
    waste collection process to enable realistic simulation of decision-making
    under uncertainty.
    """

    # Fixed input
    coords: torch.Tensor  # Depot + loc
    noisy_waste: torch.Tensor  # Observed/Noisy waste
    real_waste: torch.Tensor  # True/Real waste

    # Cost function weights
    w_waste: float
    w_length: float
    w_overflows: float

    # Maximum amount of waste before bin is considered to be overflowing
    max_waste: torch.Tensor

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
            return mask_long2bool(self.visited_, n=self.noisy_waste.size(-1))

    @property
    def dist(self):
        """Calculates the Euclidean distance matrix for the current coordinates."""
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(
            p=2, dim=-1
        )

    def __getitem__(self, key):
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
        Initializes the state from the input data.

        Args:
            input (dict): Input dictionary with 'depot', 'loc', 'noisy_waste', 'real_waste', 'max_waste'.
            edges (Tensor): Edge information for the graph.
            cost_weights (dict, optional): Weights for waste, overflows, and length costs.
            dist_matrix (Tensor, optional): Precomputed distance matrix.
            visited_dtype (torch.dtype): Data type for visited nodes (uint8 or int64 for compression).
            hrl_mask (Tensor, optional): Pre-visited nodes mask for Hierarchical RL.

        Returns:
            StateSCWCVRP: The initial state.
        """
        depot = input["depot"]
        loc = input["loc"]
        noisy_waste = input["noisy_waste"]  # Noisy/Observed
        real_waste = input["real_waste"]  # Real/Actual
        max_waste = input["max_waste"]
        batch_size, n_loc, _ = loc.size()

        # Initialize visited mask
        if visited_dtype == torch.uint8:
            visited_ = torch.zeros(
                batch_size, 1, n_loc + 1, dtype=torch.uint8, device=loc.device
            )
            if hrl_mask is not None:
                if hrl_mask.dim() == 2:
                    if hrl_mask.size(1) != n_loc:
                        print(
                            f"CRITICAL ERROR: HRL Mask width {hrl_mask.size(1)} != n_loc {n_loc}"
                        )
                    visited_[:, 0, 1:] = hrl_mask.to(torch.uint8)
                else:
                    visited_[:, 0, 1:] = hrl_mask.squeeze().to(torch.uint8)
        else:
            # Compressed mask logic (int64 bitmask)
            visited_ = torch.zeros(
                batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device
            )
            if hrl_mask is not None:
                print("Warning: HRL Mask with compressed visited mask not implemented")

        return StateSCWCVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            noisy_waste=F.pad(
                noisy_waste, (1, 0), mode="constant", value=0
            ),  # add 0 for depot
            real_waste=F.pad(
                real_waste, (1, 0), mode="constant", value=0
            ),  # add 0 for depot
            max_waste=max_waste[:, None],
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[
                :, None
            ],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=noisy_waste.new_zeros(batch_size, 1),
            visited_=visited_,
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input["depot"][:, None, :],  # Add step dimension
            cur_overflows=torch.sum(
                (input["real_waste"] >= max_waste[:, None]), dim=-1
            ),  # Use Real waste for overflows
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

    def get_final_cost(self):
        """
        Calculates the final cost of the solution once all nodes are visited.

        Returns:
            Tensor: The total calculated cost.
        """
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
        Updates the state based on the selected action (node to visit).

        Args:
            selected (Tensor): The index of the selected node (batch_size, 1).

        Returns:
            StateSCWCVRP: The updated state.
        """
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        if selected.dim() == 1:
            selected = selected[:, None]
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(
            p=2, dim=-1
        )  # (batch_dim, 1)

        # Get Real Waste for computations of capacity and overflow updates
        selected_real_waste = self.real_waste.gather(1, selected).clamp(
            max=self.max_waste
        )
        cur_total_waste = self.cur_total_waste + selected_real_waste

        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (self.used_capacity + selected_real_waste) * (
            prev_a != 0
        ).float()

        # Update the number of overflows (Using Real Waste)
        was_overflowing = self.real_waste[self.ids, selected] >= self.max_waste
        cur_overflows = self.cur_overflows - was_overflowing.float().sum(dim=-1)

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
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
        """Checks if all instances in the batch have finished their tours (returned to depot)."""
        return (self.i > 0).all() and (self.prev_a == 0).all()

    def get_finished(self):
        """Checks if all nodes have been visited in each instance."""
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        """Returns the index of the current node."""
        return self.prev_a

    def get_remaining_overflows(self):
        """Returns the number of remaining overflows."""
        return self.cur_overflows.unsqueeze(-1)

    def get_current_efficiency(self):
        """Calculates the current efficiency (collected waste per unit length)."""
        efficiency = self.cur_total_waste / self.lengths
        return torch.nan_to_num(efficiency, nan=0.0)

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible.
        """
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.noisy_waste.size(-1))

        # Check which nodes exceed vehicle capacity if visited from current state (Based on NOISY waste)
        waste_to_collect = self.noisy_waste[:, 1:].clamp(max=self.max_waste)

        # Used Capacity is REAL (updated in update step).
        # So we check: Current Real Used + Dest Noisy Waste > Capacity.
        exceeds_cap = (self.used_capacity[:, :, None] > 0) & (
            waste_to_collect[:, None, :] + self.used_capacity[:, :, None]
            > self.vehicle_capacity
        )

        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        mask_depot = torch.zeros_like(self.prev_a, dtype=torch.bool)
        has_valid_customer = ~mask_loc.all(dim=-1)
        mask_depot = mask_depot | ((self.prev_a == 0) & has_valid_customer)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1).bool()

    def get_edges_mask(self):
        """Returns a mask indicating which edges are valid from the current node."""
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
        """Returns the actions as the solution representation."""
        return actions
