"""
State representation for the Waste Collection Vehicle Routing Problem (WCVRP).
"""

import torch
import torch.nn.functional as F
from typing import NamedTuple
from logic.src.utils.boolmask import mask_long_scatter
from ..base import BaseState, refactor_state


@refactor_state
class StateWCVRP(NamedTuple):
    """
    Data class representing the state of a WCVRP tour.
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
    visited_: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    cur_total_waste: torch.Tensor
    cur_overflows: torch.Tensor
    i: torch.Tensor
    edges: torch.Tensor
    dist_matrix: torch.Tensor

    def __getitem__(self, key):
        """Indexes the state for batch slicing."""
        if torch.is_tensor(key) or isinstance(key, slice):
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                cur_total_waste=self.cur_total_waste[key],
                cur_overflows=self.cur_overflows[key],
            )
        return self[key]

    @staticmethod
    def initialize(input, edges, cost_weights=None, dist_matrix=None, visited_dtype=torch.uint8, hrl_mask=None):
        """Initializes the state for a batch of instances."""
        common = BaseState.initialize_common(input, visited_dtype)
        
        if hrl_mask is not None:
            if hrl_mask.dim() == 2:
                common["visited_"][:, 0, 1:] = hrl_mask.to(torch.uint8)
            else:
                common["visited_"][:, 0, 1:] = hrl_mask.squeeze().to(torch.uint8)

        return StateWCVRP(
            coords=common["coords"],
            waste=F.pad(input["waste"], (1, 0), mode="constant", value=0),
            max_waste=input["max_waste"][:, None],
            ids=common["ids"],
            prev_a=common["prev_a"],
            visited_=common["visited_"],
            lengths=common["lengths"],
            cur_coord=common["cur_coord"],
            cur_total_waste=torch.zeros(common["batch_size"], 1, device=input["loc"].device),
            cur_overflows=torch.sum((input["waste"] >= input["max_waste"][:, None]), dim=-1),
            i=common["i"],
            w_waste=1 if cost_weights is None else cost_weights["waste"],
            w_overflows=1 if cost_weights is None else cost_weights["overflows"],
            w_length=1 if cost_weights is None else cost_weights["length"],
            edges=edges,
            dist_matrix=dist_matrix,
        )

    def get_final_cost(self):
        """Calculates the final cost after the tour is finished."""
        assert self.all_finished()
        return (
            self.w_overflows * self.cur_overflows
            + self.w_length * self.lengths
            - self.w_waste * self.cur_total_waste
        )

    def update(self, selected):
        """Updates the state after moving to a new node."""
        assert self.i.size(0) == 1
        selected = selected[:, None]
        prev_a = selected

        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        selected_waste = self.waste[self.ids, selected].clamp(max=self.max_waste[self.ids, 0])
        cur_total_waste = self.cur_total_waste + selected_waste

        cur_overflows = self.cur_overflows - torch.sum(
            (self.waste[self.ids, selected] >= self.max_waste[self.ids, 0]), dim=-1
        )

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)

        return self._replace(
            prev_a=prev_a,
            visited_=visited_,
            lengths=lengths,
            cur_coord=cur_coord,
            cur_total_waste=cur_total_waste,
            cur_overflows=cur_overflows,
            i=self.i + 1,
        )

    def get_remaining_overflows(self):
        """Returns the count of bins currently overflowing but not yet visited."""
        return self.cur_overflows.unsqueeze(-1)

    def get_current_efficiency(self):
        """Calculates collected waste per unit distance."""
        efficiency = self.cur_total_waste / self.lengths
        return torch.nan_to_num(efficiency, nan=0.0)

    def get_mask(self):
        """Returns a mask indicating which nodes are invalid to visit next."""
        visited_ = self.visited > 0
        mask = visited_ | visited_[:, :, 0:1]
        mask[:, :, 0] = 0
        return mask
