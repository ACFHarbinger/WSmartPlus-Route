"""
IRP problem definition for offline evaluation.

Attributes:
    IRP: Inventory Routing Problem (IRP) definition.

Example:
    >>> import torch
    >>> from logic.src.envs.tasks.irp import IRP
    >>> dataset = {
    ...     "locs": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
    ...     "demand": torch.tensor([[0.0, 1.0]]),
    ...     "holding_cost": torch.tensor([[0.0, 1.0]]),
    ...     "stockout": torch.tensor([[0.0, 1.0]]),
    ...     "routing_cost": torch.tensor([2.0]),
    ...     "depot": torch.tensor([0.0]),
    ... }
    >>> pi = torch.tensor([[[0, 1, 0]]])
    >>> length, cost_dict, _ = IRP.get_costs(dataset, pi)
    >>> print(length)
    tensor([4.0])
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.envs.tasks.base import BaseProblem


class IRP(BaseProblem):
    """
    Inventory Routing Problem (IRP).

    Offline cost computation for multi-period IRP solutions.
    The cost is the sum of routing distance plus holding costs minus any
    stockout penalty savings, evaluated from a completed IRPEnv episode state.

    Attributes:
        NAME: Environment name identifier.
    """

    NAME = "irp"

    @staticmethod
    def get_costs(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        cw_dict: Optional[Dict[str, float]],
        dist_matrix: Optional[torch.Tensor] = None,
    ):
        """
        Compute IRP costs from a completed episode state dict.

        For multi-period IRP, ``dataset`` is expected to be the terminal
        TensorDict produced by IRPEnv (containing ``routing_cost``,
        ``holding_cost``, and ``stockout`` fields set by ``_get_reward``).
        If those fields are absent the method falls back to computing the
        routing component from ``pi`` alone (holding cost is then ignored).

        Args:
            dataset: Terminal TensorDict or dict with episode data.
            pi: Tour tensor [batch, steps] (unused when pre-computed costs present).
            cw_dict: Optional weight dict with keys ``"routing"``, ``"holding"``,
                     ``"stockout"`` (defaults to 1.0 each).
            dist_matrix: Unused (IRP uses Euclidean distance within the env).

        Returns:
            Tuple of (total_cost, cost_dict, None).
        """
        routing_w = 1.0 if cw_dict is None else cw_dict.get("routing", 1.0)
        holding_w = 1.0 if cw_dict is None else cw_dict.get("holding", 1.0)
        stockout_w = 1.0 if cw_dict is None else cw_dict.get("stockout", 10.0)

        # Prefer pre-computed decomposed costs stored by IRPEnv._get_reward
        if "routing_cost" in dataset and "holding_cost" in dataset and "stockout" in dataset:
            routing = dataset["routing_cost"]
            holding = dataset["holding_cost"]
            stockout = dataset["stockout"]
        else:
            # Fallback: compute routing cost from tour, ignore inventory costs
            routing = IRP.get_tour_length(dataset, pi, dist_matrix)
            holding = torch.zeros_like(routing)
            stockout = torch.zeros_like(routing)

        total = routing_w * routing + holding_w * holding + stockout_w * stockout

        return (
            total,
            {
                "routing_cost": routing,
                "holding_cost": holding,
                "stockout": stockout,
                "total": total,
            },
            None,
        )
