"""
Sampling utilities for policy execution.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import torch
import torch.nn.functional as F

from .tensors import do_batch_rep


def sample_many(
    inner_func: Callable[..., Any],
    get_cost_func: Callable[..., Any],
    input: Any,
    batch_rep: int = 1,
    iter_rep: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples many solutions by repeated execution.

    Args:
        inner_func: Function producing policy and log probabilities.
        get_cost_func: Function computing costs and mask.
        input: Input node features.
        batch_rep: Batch replication factor. Defaults to 1.
        iter_rep: Iteration replication. Defaults to 1.

    Returns:
        tuple: (min_policies, min_costs)
    """
    input = do_batch_rep(input, batch_rep)
    costs = []
    pis = []
    for _ in range(iter_rep):
        _log_p, pi = inner_func(input)

        cost, _mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)

    # Reshape and pad policies to match the maximum sequence length across all samples
    pis_cat = torch.cat([F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis], 1)
    costs_cat = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs_cat.min(-1)

    # (batch_size, minlength)
    minpis = pis_cat[torch.arange(pis_cat.size(0), out=argmincosts.new()), argmincosts]
    return minpis, mincosts
