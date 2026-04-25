"""
PCTSP problem definition for offline evaluation.

Attributes:
    PCTSP: Prize-Collecting TSP (PCTSP) definition.

Example:
    >>> import torch
    >>> from logic.src.envs.tasks.pctsp import PCTSP
    >>> dataset = {
    ...     "locs": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
    ...     "penalty": torch.tensor([[0.0, 10.0]]),
    ...     "real_prize": torch.tensor([[0.0, 10.0]]),
    ...     "depot": torch.tensor([0.0]),
    ... }
    >>> pi = torch.tensor([[[0, 1, 0]]])
    >>> length, cost_dict, _ = PCTSP.get_costs(dataset, pi)
    >>> print(length)
    tensor([-2.0])
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.envs.tasks.base import BaseProblem


class PCTSP(BaseProblem):
    """
    Prize-Collecting TSP (PCTSP).

    Objective: saved_penalties - (tour_length + remaining_penalties).

    Attributes:
        NAME: Environment name identifier.
    """

    NAME = "pctsp"

    @staticmethod
    def get_costs(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        cw_dict: Optional[Dict[str, float]],
        dist_matrix: Optional[torch.Tensor] = None,
    ):
        """
        Compute PCTSP objective from tour.

        Args:
            dataset: Dict with ``locs``, ``depot``, ``penalty`` [B, N],
                     and ``real_prize`` [B, N].
            pi: Tour tensor [B, T].
            cw_dict: Unused.
            dist_matrix: Optional.

        Returns:
            Tuple of (neg_reward, cost_dict, None).
        """
        if pi.size(-1) <= 1:
            penalty_total = (
                dataset["penalty"].sum(-1) if "penalty" in dataset else torch.zeros(pi.size(0), device=pi.device)
            )
            return (
                penalty_total,
                {
                    "saved_penalty": torch.zeros_like(penalty_total),
                    "length": torch.zeros_like(penalty_total),
                    "total": penalty_total,
                },
                None,
            )

        # Pad penalty / prize with depot=0 if needed
        penalty = dataset["penalty"]  # [B, N] or [B, N+1]
        if penalty.shape[-1] == pi.max().item():
            penalty = torch.cat([torch.zeros_like(penalty[:, :1]), penalty], dim=-1)

        length = PCTSP.get_tour_length(dataset, pi, dist_matrix)
        saved_penalty = penalty.gather(1, pi).sum(-1)
        remaining_penalty = penalty[:, 1:].sum(-1) - saved_penalty

        reward = saved_penalty - (length + remaining_penalty)
        return -reward, {"saved_penalty": saved_penalty, "length": length, "total": -reward}, None
