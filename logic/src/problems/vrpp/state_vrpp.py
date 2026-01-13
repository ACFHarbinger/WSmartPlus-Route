"""
State representation for the Vehicle Routing Problem with Profits (VRPP).
"""

import torch
import torch.nn.functional as F
from typing import NamedTuple
from logic.src.utils.boolmask import mask_long_scatter
from ..base import BaseState, refactor_state


@refactor_state
class StateVRPP(NamedTuple):
    """
    Data class representing the state of a VRPP tour.
    """

    # Fixed input
    coords: torch.Tensor
    waste: torch.Tensor
    profit_vars: dict
    max_waste: torch.Tensor
    ids: torch.Tensor

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    cur_total_waste: torch.Tensor
    cur_negative_profit: torch.Tensor
    i: torch.Tensor
    edges: torch.Tensor
    dist_matrix: torch.Tensor
    w_waste: float
    w_length: float

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
                cur_negative_profit=self.cur_negative_profit[key],
            )
        return self[key]

    @staticmethod
    def initialize(input, edges, profit_vars=None, cost_weights=None, dist_matrix=None, visited_dtype=torch.uint8, **kwargs):
        """Initializes the state for a batch of instances."""
        common = BaseState.initialize_common(input, visited_dtype)
        
        if profit_vars is None:
            profit_vars = {"cost_km": 1.0, "revenue_kg": 1.0}

        return StateVRPP(
            coords=common["coords"],
            waste=F.pad(input["waste"], (1, 0), mode="constant", value=0),
            profit_vars=profit_vars,
            max_waste=input["max_waste"][:, None],
            ids=common["ids"],
            prev_a=common["prev_a"],
            visited_=common["visited_"],
            lengths=common["lengths"],
            cur_coord=common["cur_coord"],
            cur_total_waste=torch.zeros(common["batch_size"], 1, device=input["loc"].device),
            cur_negative_profit=torch.zeros(common["batch_size"], 1, device=input["loc"].device),
            i=common["i"],
            edges=edges,
            dist_matrix=dist_matrix,
            w_waste=1 if cost_weights is None else cost_weights["waste"],
            w_length=1 if cost_weights is None else cost_weights["length"],
        )

    def get_final_cost(self):
        """Returns the current negative profit as the final cost."""
        assert self.all_finished()
        return self.cur_negative_profit

    def update(self, selected):
        """Updates the state after moving to a new node, calculating negative profit."""
        assert self.i.size(0) == 1
        selected = selected[:, None]
        prev_a = selected

        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        selected_waste = self.waste[self.ids, selected].clamp(max=self.max_waste)
        cur_total_waste = self.cur_total_waste + selected_waste

        cur_negative_profit = (
            self.w_length * lengths * self.profit_vars["cost_km"]
            - self.w_waste * cur_total_waste * self.profit_vars["revenue_kg"]
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
            cur_negative_profit=cur_negative_profit,
            i=self.i + 1,
        )

    def get_mask(self):
        """Returns a mask indicating which nodes are invalid to visit next."""
        visited_ = self.visited > 0
        mask = visited_ | visited_[:, :, 0:1]
        mask[:, :, 0] = 0
        return mask

    def get_current_profit(self):
        """Returns the current profit (negative cost)."""
        return -self.cur_negative_profit
