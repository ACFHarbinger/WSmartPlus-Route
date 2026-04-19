"""
PDP problem definition for offline evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.envs.tasks.base import BaseProblem


class PDP(BaseProblem):
    """
    Pickup and Delivery Problem (PDP).

    Cost = total tour length (depot implicitly added at start and end).
    """

    NAME = "pdp"

    @staticmethod
    def get_costs(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        cw_dict: Optional[Dict[str, float]],
        dist_matrix: Optional[torch.Tensor] = None,
    ):
        """
        Compute PDP tour length.

        Args:
            dataset: Dict with ``locs`` [B, N+1, 2] (depot at index 0)
                     and ``depot`` [B, 2].
            pi: Tour tensor [B, N] (customer nodes only; no depot entries).
            cw_dict: Unused.
            dist_matrix: Optional.

        Returns:
            Tuple of (tour_length, cost_dict, None).
        """
        if pi.size(-1) <= 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return z, {"length": z, "total": z}, None

        length = PDP.get_tour_length(dataset, pi, dist_matrix)
        return length, {"length": length, "total": length}, None
