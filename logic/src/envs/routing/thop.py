"""
Thief Orienteering Problem (ThOP) environment.

The agent starts at a depot, selects items to steal from customer cities,
and must return to the depot within a time budget.  The travel speed degrades
linearly with the current knapsack weight, so heavier loads slow the thief
down and consume more of the time budget per unit distance.

Objective: Maximise total profit of collected items.

Action space
------------
Action 0        : return to depot (terminates the episode).
Actions 1 … m   : pick item k (1-indexed); the thief travels to that item's
                  city, collects the item, and continues from there.

Reward: total profit of all picked items (non-negative).
Done:   thief returns to depot (action == 0, step > 0).

Speed model (Chagas & Wagner 2020)
-----------------------------------
    v(w) = v_max - (w / W) * (v_max - v_min)

where w is the current carried weight and W is the knapsack capacity.
Travel time from city i to j: t(i,j) = dist(i,j) / v(w_current).
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.generators.thop import ThOPGenerator


class ThOPEnv(RL4COEnvBase):
    """
    Thief Orienteering Problem (ThOP) environment.

    Node layout in ``td["locs"]``:
      index 0        = depot
      indices 1…N    = customer cities (N = num_loc)

    Item layout:
      ``td["item_weights"][b, k]``  — weight of item k in batch b
      ``td["item_profits"][b, k]``  — profit of item k in batch b
      ``td["item_city"][b, k]``     — city (1-based) where item k is located

    The action mask has ``num_items + 1`` entries:
      mask[0]     — depot return (always valid after step > 0)
      mask[1..m]  — item k (valid when not yet picked, capacity allows,
                             and time round-trip is feasible)
    """

    NAME: str = "thop"
    name: str = "thop"
    # Items are the primary nodes; each has (x, y, weight, profit) = 4 features
    node_dim: int = 4

    def __init__(
        self,
        generator: Optional[ThOPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        generator_params = generator_params or {}
        if generator is None:
            generator = ThOPGenerator(**generator_params, device=device)
        super().__init__(generator, generator_params, device, **kwargs)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialise ThOP episode state from a generated instance."""
        if "item_picked" in td.keys():
            return td

        device = td.device
        bs = td.batch_size
        m = td["item_weights"].shape[-1]  # total number of items

        # Thief starts at depot (city 0), nothing picked, no time elapsed
        td["current_city"] = torch.zeros(*bs, dtype=torch.long, device=device)
        td["current_time"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["current_weight"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["item_picked"] = torch.zeros(*bs, m, dtype=torch.bool, device=device)

        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["reward"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["terminated"] = torch.zeros(*bs, dtype=torch.bool, device=device)
        td["truncated"] = torch.zeros(*bs, dtype=torch.bool, device=device)

        return td

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step(self, td: TensorDict) -> TensorDict:
        return OpsMixin._step(self, td)

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Execute one ThOP action.

        Action 0  → return to depot.
        Action k  → pick item k-1 (0-indexed); travel to its city first.

        Travel time uses the speed model v(w) = v_max - (w/W)*(v_max-v_min)
        evaluated at the weight *before* the item is added to the knapsack.
        """
        action = td["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)

        # Whether this action returns to the depot
        at_depot_action = action == 0  # [B]

        # 0-indexed item to pick (clamped so depot action doesn't produce -1)
        item_idx = (action - 1).clamp(min=0)  # [B]

        # Resolve target city: item's city, or city 0 for depot return
        item_city = td["item_city"]  # [B, m]
        target_city_item = item_city.gather(-1, item_idx.unsqueeze(-1)).squeeze(-1)  # [B]
        target_city = torch.where(at_depot_action, torch.zeros_like(target_city_item), target_city_item)

        # Coordinates
        locs = td["locs"]  # [B, n+1, 2]
        current_city = td["current_city"]  # [B]

        # Current and target location: [B, 2]
        curr_loc = locs.gather(1, current_city[:, None, None].expand(-1, 1, 2)).squeeze(1)
        tgt_loc = locs.gather(1, target_city[:, None, None].expand(-1, 1, 2)).squeeze(1)

        dist = (tgt_loc - curr_loc).norm(p=2, dim=-1)  # [B]

        # Speed at current weight (before adding new item)
        capacity = td["capacity"]  # [B]
        v_max = td["v_max"]  # [B]
        v_min = td["v_min"]  # [B]
        current_weight = td["current_weight"]  # [B]
        speed = (v_max - (current_weight / capacity.clamp(min=1e-8)) * (v_max - v_min)).clamp(min=1e-8)

        time_cost = dist / speed  # [B]

        # Update elapsed time and position
        td["current_time"] = td["current_time"] + time_cost
        td["current_city"] = target_city

        # Update weight: only when picking an item (not depot return)
        item_weight_here = td["item_weights"].gather(-1, item_idx.unsqueeze(-1)).squeeze(-1)  # [B]
        weight_delta = torch.where(at_depot_action, torch.zeros_like(item_weight_here), item_weight_here)
        td["current_weight"] = current_weight + weight_delta

        # Mark the selected item as picked (only for non-depot actions)
        one_hot = torch.zeros_like(td["item_picked"]).scatter(-1, item_idx.unsqueeze(-1), True)
        td["item_picked"] = td["item_picked"] | (one_hot & ~at_depot_action.unsqueeze(-1))

        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    # ------------------------------------------------------------------
    # Done / Mask / Reward
    # ------------------------------------------------------------------

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Episode ends when the thief returns to the depot (city 0) after ≥1 step."""
        i_raw = td.get("i", torch.zeros(td.batch_size, dtype=torch.long, device=td.device))
        step = i_raw.squeeze(-1) if i_raw.dim() > 1 else i_raw
        return (td["current_city"] == 0) & (step > 0)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute the valid-action mask of shape ``[B, num_items + 1]``.

        Item k (action k, 1-indexed) is valid when ALL of:
          (a) not yet picked,
          (b) adding its weight would not exceed knapsack capacity,
          (c) time to travel to its city + time to return to depot after
              picking it fits within the remaining time budget.

        The depot (action 0) is always valid once the episode has started
        (step > 0), and is forced valid if no items can be selected.
        """
        locs = td["locs"]  # [B, n+1, 2]
        item_city = td["item_city"]  # [B, m]
        item_weights = td["item_weights"]  # [B, m]
        item_picked = td["item_picked"]  # [B, m]
        capacity = td["capacity"]  # [B]
        max_time = td["max_time"]  # [B]
        v_max = td["v_max"]  # [B]
        v_min = td["v_min"]  # [B]
        current_city = td["current_city"]  # [B]
        current_weight = td["current_weight"]  # [B]
        current_time = td["current_time"]  # [B]

        # "i" is added by OpsMixin._reset *after* _get_action_mask is first called,
        # so we must default to 0 when it is absent (initial reset call).
        i_raw = td.get("i", torch.zeros(td.batch_size, dtype=torch.long, device=td.device))
        step = i_raw.squeeze(-1) if i_raw.dim() > 1 else i_raw

        m = item_weights.shape[-1]

        # --- Current location: [B, 2] ---
        curr_loc = locs.gather(1, current_city[:, None, None].expand(-1, 1, 2)).squeeze(1)

        # --- City coordinate for every item: [B, m, 2] ---
        item_city_locs = locs.gather(1, item_city.unsqueeze(-1).expand(-1, m, 2))

        # --- Depot coordinate: [B, 1, 2] ---
        depot_loc = locs[:, 0:1, :]

        # (a) Speed at current weight: [B] → broadcast to [B, 1]
        speed_now = (
            (v_max - (current_weight / capacity.clamp(min=1e-8)) * (v_max - v_min)).clamp(min=1e-8).unsqueeze(-1)
        )  # [B, 1]

        # (b) Distance from current position to each item city: [B, m]
        dist_to_items = (item_city_locs - curr_loc.unsqueeze(1)).norm(p=2, dim=-1)

        # (c) Time to reach each item city: [B, m]
        time_to_items = dist_to_items / speed_now

        # (d) Weight after picking each item: [B, m]
        weight_after = current_weight.unsqueeze(-1) + item_weights

        # (e) Speed after picking each item: [B, m]
        speed_after = (
            v_max.unsqueeze(-1)
            - (weight_after / capacity.unsqueeze(-1).clamp(min=1e-8)) * (v_max - v_min).unsqueeze(-1)
        ).clamp(min=1e-8)

        # (f) Distance from each item city back to the depot: [B, m]
        dist_items_to_depot = (item_city_locs - depot_loc).norm(p=2, dim=-1)

        # (g) Return travel time after picking each item: [B, m]
        time_return = dist_items_to_depot / speed_after

        # (h) Remaining time budget: [B, 1]
        remaining = (max_time - current_time).unsqueeze(-1)

        # --- Validity conditions for each item ---
        time_feasible = (time_to_items + time_return) <= remaining + 1e-6  # [B, m]
        under_cap = weight_after <= capacity.unsqueeze(-1) + 1e-6  # [B, m]
        item_mask = ~item_picked & under_cap & time_feasible  # [B, m]

        # --- Depot validity ---
        # Forced valid if step > 0, or if literally no item can be picked
        no_items_left = ~item_mask.any(dim=-1)  # [B]
        depot_valid = (step > 0) | no_items_left  # [B]

        # Full mask: [B, m+1]
        return torch.cat([depot_valid.unsqueeze(-1), item_mask], dim=-1)

    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Total profit of all items collected during the episode."""
        return (td["item_profits"] * td["item_picked"].float()).sum(dim=-1)
