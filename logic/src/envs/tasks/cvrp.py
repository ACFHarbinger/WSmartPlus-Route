"""
CVRP problem definition for offline evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.envs.tasks.base import BaseProblem


class CVRP(BaseProblem):
    """
    Capacitated Vehicle Routing Problem (CVRP).

    Validates capacity constraints and computes total tour length.
    """

    NAME = "cvrp"

    @staticmethod
    def get_costs(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        cw_dict: Optional[Dict[str, float]],
        dist_matrix: Optional[torch.Tensor] = None,
    ):
        """
        Compute CVRP tour length.

        Args:
            dataset: Dict / TensorDict with ``locs``, ``depot``, ``demand``,
                     and ``vehicle_capacity``.
            pi: Tour tensor [B, T] (may contain repeated depot visits as 0).
            cw_dict: Unused.
            dist_matrix: Optional distance matrix.

        Returns:
            Tuple of (tour_length, cost_dict, None).
        """
        CVRP.validate_tours(pi)
        if pi.size(-1) == 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return z, {"length": z, "total": z}, None

        length = CVRP.get_tour_length(dataset, pi, dist_matrix)
        return length, {"length": length, "total": length}, None
