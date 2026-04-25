"""
ThOP problem definition for offline evaluation.

Evaluates a complete ThOP solution given as a sequence of actions:
  0       → depot return (terminates the tour)
  1 … m   → pick item k (1-indexed)

The evaluation simulates the speed-degraded travel and returns the total
profit together with feasibility metrics (time used, weight used).

Attributes:
    ThOP: Thief Orienteering Problem (ThOP) definition.

Example:
    >>> import torch
    >>> from logic.src.envs.tasks.thop import ThOP
    >>> dataset = {
    ...     "locs": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
    ...     "item_weights": torch.tensor([[0.0, 1.0]]),
    ...     "item_profits": torch.tensor([[0.0, 10.0]]),
    ...     "item_city": torch.tensor([[0.0, 1.0]]),
    ...     "capacity": torch.tensor([1.0]),
    ...     "max_time": torch.tensor([10.0]),
    ...     "v_max": torch.tensor([1.0]),
    ...     "v_min": torch.tensor([0.1]),
    ...     "depot": torch.tensor([0.0]),
    ... }
    >>> pi = torch.tensor([[[0, 1, 0]]])
    >>> length, cost_dict, _ = ThOP.get_costs(dataset, pi)
    >>> print(length)
    tensor([-2.0])
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from logic.src.envs.tasks.base import BaseProblem


class ThOP(BaseProblem):
    """
    Thief Orienteering Problem (ThOP).

    Combines route profit maximisation (Orienteering Problem) with item
    selection under a knapsack capacity constraint.  Travel speed decreases
    linearly with the current carried weight:

        v(w) = v_max - (w / W) * (v_max - v_min)

    so heavier loads consume more of the time budget per kilometre.

    Objective: maximise total item profit subject to
      * total travel time ≤ T_max (time budget), and
      * total carried weight ≤ W (knapsack capacity).

    Attributes:
        NAME: Environment name identifier.
    """

    NAME = "thop"

    @staticmethod
    def get_costs(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        cw_dict: Optional[Dict[str, Any]],
        dist_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], None]:
        """
        Evaluate a ThOP solution.

        Args:
            dataset: Problem instance TensorDict (or dict) with keys:
                ``locs`` ``[B, n+1, 2]``, ``item_weights`` ``[B, m]``,
                ``item_profits`` ``[B, m]``, ``item_city`` ``[B, m]``
                (1-based), ``capacity`` ``[B]``, ``max_time`` ``[B]``,
                ``v_max`` ``[B]``, ``v_min`` ``[B]``.
            pi: Action sequence ``[B, T]`` where 0 = depot and 1…m are
                1-indexed item picks.  The tour ends at the first 0 after
                step > 0, or at the last element if no explicit return.
            cw_dict: Optional weight overrides (currently unused; kept for
                interface parity with other tasks).
            dist_matrix: Ignored – ThOP uses Euclidean distances from
                ``locs``.

        Returns:
            ``(-total_profit, info_dict, None)`` where ``info_dict`` has:
            ``profit``, ``time_used``, ``weight_used``, ``time_feasible``,
            ``weight_feasible``, ``feasible``, ``total``.
        """
        if pi.dim() == 1:
            pi = pi.unsqueeze(0)

        B, T = pi.shape
        device = pi.device

        locs = dataset["locs"]  # [B, n+1, 2]
        item_weights = dataset["item_weights"]  # [B, m]
        item_profits = dataset["item_profits"]  # [B, m]
        item_city = dataset["item_city"]  # [B, m]  (1-based)
        capacity = dataset["capacity"]  # [B]
        max_time = dataset["max_time"]  # [B]
        v_max = dataset["v_max"]  # [B]
        v_min = dataset["v_min"]  # [B]

        # Running state
        current_city = torch.zeros(B, dtype=torch.long, device=device)  # depot
        current_weight = torch.zeros(B, dtype=torch.float32, device=device)
        current_time = torch.zeros(B, dtype=torch.float32, device=device)
        total_profit = torch.zeros(B, dtype=torch.float32, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(T):
            action = pi[:, t]  # [B]
            still_active = ~done  # [B]

            at_depot = action == 0  # [B]
            item_idx = (action - 1).clamp(min=0)  # [B]  (0-indexed)

            # Target city
            target_city_item = item_city.gather(-1, item_idx.unsqueeze(-1)).squeeze(-1)  # [B]
            target_city = torch.where(at_depot, torch.zeros_like(target_city_item), target_city_item)

            # Distance
            curr_loc = locs.gather(1, current_city[:, None, None].expand(B, 1, 2)).squeeze(1)
            tgt_loc = locs.gather(1, target_city[:, None, None].expand(B, 1, 2)).squeeze(1)
            dist = (tgt_loc - curr_loc).norm(p=2, dim=-1)  # [B]

            # Speed at current weight (before pickup)
            speed = (v_max - (current_weight / capacity.clamp(min=1e-8)) * (v_max - v_min)).clamp(min=1e-8)
            time_cost = dist / speed  # [B]

            # Accumulate time and update position for active instances
            current_time = torch.where(still_active, current_time + time_cost, current_time)
            current_city = torch.where(still_active, target_city, current_city)

            # Collect item profit and weight (only for item actions, not depot)
            item_w_here = item_weights.gather(-1, item_idx.unsqueeze(-1)).squeeze(-1)  # [B]
            item_p_here = item_profits.gather(-1, item_idx.unsqueeze(-1)).squeeze(-1)  # [B]

            picking = still_active & ~at_depot
            current_weight = torch.where(picking, current_weight + item_w_here, current_weight)
            total_profit = torch.where(picking, total_profit + item_p_here, total_profit)

            # Mark done when returning to depot (action=0, t>0)
            done = done | (still_active & at_depot & (t > 0))

        # Force return-to-depot for sequences that never explicitly returned
        not_returned = ~done
        if not_returned.any():
            curr_loc = locs.gather(1, current_city[:, None, None].expand(B, 1, 2)).squeeze(1)
            depot_loc = locs[:, 0, :]
            dist_home = (depot_loc - curr_loc).norm(p=2, dim=-1)
            speed = (v_max - (current_weight / capacity.clamp(min=1e-8)) * (v_max - v_min)).clamp(min=1e-8)
            time_home = dist_home / speed
            current_time = torch.where(not_returned, current_time + time_home, current_time)

        # Feasibility checks
        time_feasible = current_time <= max_time + 1e-6  # [B]
        weight_feasible = current_weight <= capacity + 1e-6  # [B]
        feasible = time_feasible & weight_feasible

        # Negative profit as cost (consistent with minimisation convention)
        neg_profit = -total_profit

        info: Dict[str, Any] = {
            "profit": total_profit,
            "time_used": current_time,
            "weight_used": current_weight,
            "time_feasible": time_feasible,
            "weight_feasible": weight_feasible,
            "feasible": feasible,
            "total": neg_profit,
        }
        return (neg_profit, info, None)
