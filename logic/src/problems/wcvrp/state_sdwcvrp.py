"""
State representation for the Split Delivery Waste Collection Vehicle Routing Problem (SDWCVRP).
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F

from logic.src.utils.definitions import VEHICLE_CAPACITY

from ..base import BaseState, refactor_state


@refactor_state
class StateSDWCVRP(NamedTuple):
    """
    Data class representing the state of an SDWCVRP tour.
    """

    # Fixed input
    coords: torch.Tensor
    waste: torch.Tensor
    w_waste: float
    w_length: float
    w_overflows: float
    max_waste: torch.Tensor
    ids: torch.Tensor

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    demands_with_depot: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    cur_overflows: torch.Tensor
    cur_total_waste: torch.Tensor
    i: torch.Tensor
    edges: torch.Tensor
    dist_matrix: torch.Tensor
    vehicle_capacity: float

    def __getitem__(self, key):
        """Indexes the state for batch slicing."""
        if torch.is_tensor(key) or isinstance(key, slice):
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                demands_with_depot=self.demands_with_depot[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                cur_overflows=self.cur_overflows[key],
                cur_total_waste=self.cur_total_waste[key],
            )
        return self[key]

    @property
    def visited(self):
        """In SDWCVRP, visited is defined as nodes with no waste left."""
        return (self.demands_with_depot <= 1e-5).to(torch.uint8)

    @staticmethod
    def initialize(input, edges, cost_weights=None, dist_matrix=None, **kwargs):
        """Initializes the state for a batch of instances."""
        common = BaseState.initialize_common(input)

        return StateSDWCVRP(
            coords=common["coords"],
            waste=input["waste"],
            max_waste=input["max_waste"][:, None],
            ids=common["ids"],
            prev_a=common["prev_a"],
            used_capacity=input["waste"].new_zeros(common["batch_size"], 1),
            demands_with_depot=F.pad(input["waste"], (1, 0), mode="constant", value=0),
            lengths=common["lengths"],
            cur_coord=common["cur_coord"],
            cur_overflows=torch.sum((input["waste"] >= input["max_waste"][:, None]), dim=-1),
            cur_total_waste=torch.zeros(common["batch_size"], 1, device=input["loc"].device),
            i=common["i"],
            w_waste=1 if cost_weights is None else cost_weights["waste"],
            w_overflows=1 if cost_weights is None else cost_weights["overflows"],
            w_length=1 if cost_weights is None else cost_weights["length"],
            edges=edges,
            dist_matrix=dist_matrix,
            vehicle_capacity=VEHICLE_CAPACITY,
        )

    def get_final_cost(self):
        """Calculates the final cost after the tour is finished."""
        assert self.all_finished()
        length_cost = self.w_length * self.lengths + self.w_length * (
            self.coords[self.ids, 0, :] - self.cur_coord
        ).norm(p=2, dim=-1)
        return self.w_overflows * self.cur_overflows + length_cost + self.w_waste * self.cur_total_waste

    def update(self, selected):
        """Updates the state after moving to a new node, supporting partial collection."""
        assert self.i.size(0) == 1
        device = self.coords.device
        selected = selected[:, None]
        prev_a = selected

        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        batch_size = selected.size(0)
        rng = torch.arange(batch_size, device=device)[:, None]

        remaining_cap = self.vehicle_capacity - self.used_capacity
        d = torch.min(self.demands_with_depot[rng, selected], remaining_cap)

        actual_collected = d.clone()
        used_capacity = (self.used_capacity + actual_collected) * (prev_a != 0).float()

        demands_with_depot = self.demands_with_depot.clone()
        demands_with_depot[rng, selected] -= actual_collected
        cur_total_waste = self.cur_total_waste + actual_collected
        cur_overflows = torch.sum(demands_with_depot[:, 1:] >= self.max_waste, dim=-1)

        return self._replace(
            prev_a=prev_a,
            used_capacity=used_capacity,
            demands_with_depot=demands_with_depot,
            lengths=lengths,
            cur_coord=cur_coord,
            cur_overflows=cur_overflows,
            cur_total_waste=cur_total_waste,
            i=self.i + 1,
        )

    def get_remaining_overflows(self):
        """Returns the count of bins currently overflowing but not yet visited (waste > 0)."""
        return self.cur_overflows.unsqueeze(-1)

    def get_current_efficiency(self):
        """Calculates collected waste per unit distance."""
        efficiency = self.cur_total_waste / self.lengths
        return torch.nan_to_num(efficiency, nan=0.0)

    def get_mask(self):
        """Returns a mask indicating which nodes are invalid to visit next, accounting for partial deliveries."""
        at_capacity = self.used_capacity >= self.vehicle_capacity - 1e-5
        mask_loc = (self.demands_with_depot[:, None, 1:] <= 1e-5) | at_capacity[:, :, None]
        mask_depot = torch.zeros_like(self.prev_a, dtype=torch.bool)
        has_valid_customer = ~mask_loc.all(dim=-1)
        mask_depot = mask_depot | ((self.prev_a == 0) & has_valid_customer)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1).bool()
