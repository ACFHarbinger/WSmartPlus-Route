"""
ATSP problem definition for offline evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.envs.tasks.base import BaseProblem


class ATSP(BaseProblem):
    """
    Asymmetric Traveling Salesman Problem (ATSP).

    Cost = total asymmetric tour length (including closing edge).
    """

    NAME = "atsp"

    @staticmethod
    def get_costs(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        cw_dict: Optional[Dict[str, float]],
        dist_matrix: Optional[torch.Tensor] = None,
    ):
        """
        Compute ATSP tour cost from a cost matrix and tour sequence.

        Args:
            dataset: Dict / TensorDict with ``cost_matrix`` [B, N, N].
            pi: Tour tensor [B, N] — all N nodes in visit order.
            cw_dict: Unused (single objective).
            dist_matrix: Ignored; uses ``dataset["cost_matrix"]``.

        Returns:
            Tuple of (cost, cost_dict, None).
        """
        cost_matrix = dataset["cost_matrix"]  # [B, N, N]
        bs = cost_matrix.shape[0]

        nodes_src = pi  # [B, N]
        nodes_tgt = torch.roll(pi, -1, dims=-1)  # closing edge included
        batch_idx = torch.arange(bs, device=cost_matrix.device).unsqueeze(1)
        cost = cost_matrix[batch_idx, nodes_src, nodes_tgt].sum(-1)  # [B]

        return cost, {"length": cost, "total": cost}, None
