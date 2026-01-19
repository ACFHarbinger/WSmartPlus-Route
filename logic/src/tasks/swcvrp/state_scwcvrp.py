"""
State representation for the Stochastic Capacitated Waste Collection Vehicle Routing Problem (SCWCVRP).
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F

from logic.src.utils.definitions import VEHICLE_CAPACITY
from logic.src.utils.functions.boolmask import mask_long2bool, mask_long_scatter

from ..base import BaseState, refactor_state


@refactor_state
class StateSCWCVRP(NamedTuple):
    """
    Data class representing the state of an SCWCVRP tour.
    """

    # Fixed input
    coords: torch.Tensor
    real_waste: torch.Tensor
    noisy_waste: torch.Tensor
    w_waste: float
    w_length: float
    w_overflows: float
    max_waste: torch.Tensor
    ids: torch.Tensor

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor
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
        **kwargs,
    ):
        """Initializes the state for a batch of instances, using noisy waste for observability."""
        common = BaseState.initialize_common(input, visited_dtype)

        return StateSCWCVRP(
            coords=common["coords"],
            real_waste=F.pad(input["real_waste"], (1, 0), mode="constant", value=0),
            noisy_waste=F.pad(input["noisy_waste"], (1, 0), mode="constant", value=0),
            max_waste=input["max_waste"][:, None],
            ids=common["ids"],
            prev_a=common["prev_a"],
            used_capacity=input["real_waste"].new_zeros(common["batch_size"], 1),
            visited_=common["visited_"],
            lengths=common["lengths"],
            cur_coord=common["cur_coord"],
            cur_overflows=torch.sum((input["real_waste"] >= input["max_waste"][:, None]), dim=-1),
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
        """Calculates the final cost based on real waste levels."""
        assert self.all_finished()
        length_cost = self.w_length * self.lengths + self.w_length * (
            self.coords[self.ids, 0, :] - self.cur_coord
        ).norm(p=2, dim=-1)
        return self.w_overflows * self.cur_overflows + length_cost + self.w_waste * self.cur_total_waste

    def update(self, selected):
        """Updates the state after moving to a new node, using real waste for transitions."""
        assert self.i.size(0) == 1
        selected = selected[:, None]
        prev_a = selected

        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        remaining_cap = self.vehicle_capacity - self.used_capacity
        d = self.real_waste[self.ids, selected].clamp(max=self.max_waste)

        actual_collected = d.clone()
        is_node = prev_a != 0
        violation_mask = is_node & (d > remaining_cap)
        actual_collected[violation_mask] = remaining_cap[violation_mask]

        cur_total_waste = self.cur_total_waste + actual_collected
        used_capacity = (self.used_capacity + actual_collected) * is_node.float()

        cur_overflows = self.cur_overflows - torch.sum((self.real_waste[self.ids, selected] >= self.max_waste), dim=-1)

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)

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

    def get_remaining_overflows(self):
        """Returns the count of bins currently overflowing based on real waste."""
        return self.cur_overflows.unsqueeze(-1)

    @property
    def waste(self):
        """Standard key for observable waste level."""
        return self.noisy_waste

    def get_current_efficiency(self):
        """Calculates collected waste (real) per unit distance."""
        efficiency = self.cur_total_waste / self.lengths
        return torch.nan_to_num(efficiency, nan=0.0)

    def get_mask(self):
        """Returns a mask based on noisy (estimated) waste levels."""
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.noisy_waste.size(-1))

        waste_estimated = self.noisy_waste[:, 1:].clamp(max=self.max_waste)
        exceeds_cap = (self.used_capacity[:, :, None] > 0) & (
            waste_estimated[:, None, :] + self.used_capacity[:, :, None] > self.vehicle_capacity
        )

        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap
        mask_depot = torch.zeros_like(self.prev_a, dtype=torch.bool)
        has_valid_customer = ~mask_loc.all(dim=-1)
        mask_depot = mask_depot | ((self.prev_a == 0) & has_valid_customer)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1).bool()
