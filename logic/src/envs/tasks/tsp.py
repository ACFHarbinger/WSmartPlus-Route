"""
TSP problem definition for offline evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from logic.src.envs.tasks.base import BaseProblem


class TSP(BaseProblem):
    """
    Traveling Salesman Problem (TSP).

    Objective: Minimise total tour length (visit every node exactly once,
    return to start).  No depot — the tour is a Hamiltonian cycle over all
    nodes.

    Cost computation supports both Euclidean (``locs`` key) and
    pre-computed distance matrix inputs.
    """

    NAME = "tsp"

    @staticmethod
    def get_costs(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        cw_dict: Optional[Dict[str, Any]],
        dist_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], None]:
        """
        Compute TSP tour cost.

        Args:
            dataset: Problem data containing ``locs`` or ``cost_matrix``.
            pi: Tour indices of shape ``(batch, num_nodes)``.
            cw_dict: Unused for TSP; kept for interface compatibility.
            dist_matrix: Optional pre-computed distance matrix ``(B, N, N)``
                or ``(N, N)``.

        Returns:
            Tuple of ``(tour_length, {"length": tour_length, "total": tour_length}, None)``.
        """
        if pi.size(-1) <= 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return (z, {"length": z, "total": z}, None)

        if dist_matrix is not None and isinstance(dist_matrix, torch.Tensor):
            length = TSP._cost_from_matrix(pi, dist_matrix)
        elif "cost_matrix" in dataset:
            length = TSP._cost_from_matrix(pi, dataset["cost_matrix"])
        else:
            length = TSP._cost_from_locs(pi, dataset["locs"])

        return (length, {"length": length, "total": length}, None)

    @staticmethod
    def _cost_from_locs(pi: torch.Tensor, locs: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean tour length including the closing edge."""
        # pi: (B, N), locs: (B, N, 2)
        visited = locs.gather(1, pi.unsqueeze(-1).expand(*pi.shape, locs.shape[-1]))
        # Closing edge: last node → first node
        closed = torch.cat([visited, visited[:, :1]], dim=1)
        return (closed[:, 1:] - closed[:, :-1]).norm(dim=-1).sum(dim=-1)

    @staticmethod
    def _cost_from_matrix(pi: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        """Compute tour length from a (possibly asymmetric) cost matrix."""
        B, N = pi.shape
        if mat.dim() == 2:
            mat = mat.unsqueeze(0).expand(B, -1, -1)
        # Closed tour: append first node to close the cycle
        pi_closed = torch.cat([pi, pi[:, :1]], dim=1)
        src = pi_closed[:, :-1]  # (B, N)
        dst = pi_closed[:, 1:]  # (B, N)
        batch_idx = torch.arange(B, device=pi.device).unsqueeze(1).expand(B, N)
        return mat[batch_idx, src, dst].sum(dim=-1)
