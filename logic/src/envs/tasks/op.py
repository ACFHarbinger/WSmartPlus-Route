"""
OP problem definition for offline evaluation.

Attributes:
    OP: Orienteering Problem (OP) definition.

Example:
    >>> import torch
    >>> from logic.src.envs.tasks.op import OP
    >>> dataset = {
    ...     "locs": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
    ...     "prize": torch.tensor([[0.0, 10.0]]),
    ...     "time_limit": torch.tensor([10.0]),
    ...     "depot": torch.tensor([0.0]),
    ... }
    >>> pi = torch.tensor([[[0, 1, 0]]])
    >>> length, cost_dict, _ = OP.get_costs(dataset, pi)
    >>> print(length)
    tensor([-2.0])
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.envs.tasks.base import BaseProblem


class OP(BaseProblem):
    """
    Orienteering Problem (OP).

    Reward = total collected prize (maximisation).
    Returned as negative cost for minimisation frameworks.

    Attributes:
        NAME: Environment name identifier.
    """

    NAME = "op"

    @staticmethod
    def get_costs(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        cw_dict: Optional[Dict[str, float]],
        dist_matrix: Optional[torch.Tensor] = None,
    ):
        """
        Compute OP prize and tour length.

        Args:
            dataset: Dict with ``prize`` [B, N] and ``locs``/``depot``.
            pi: Tour tensor [B, T] (may include depot = 0 as last node).
            cw_dict: Unused.
            dist_matrix: Optional.

        Returns:
            Tuple of (neg_prize, cost_dict, None).
        """
        if pi.size(-1) <= 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return z, {"prize": z, "length": z, "total": z}, None

        # Prize: depot has zero prize (index 0), customers in [1..N]
        prize_all = dataset["prize"]  # [B, N] or [B, N+1] with depot=0
        if prize_all.shape[-1] == pi.shape[-1] - 1:
            prize_all = torch.cat([torch.zeros_like(prize_all[:, :1]), prize_all], dim=-1)

        collected_prize = prize_all.gather(1, pi).sum(-1)
        length = OP.get_tour_length(dataset, pi, dist_matrix)

        neg_prize = -collected_prize
        return neg_prize, {"prize": collected_prize, "length": length, "total": neg_prize}, None
