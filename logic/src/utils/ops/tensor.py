"""
General Tensor Manipulation Operations.
"""

from __future__ import annotations

import torch


def unbatchify_and_gather(
    x: torch.Tensor,
    idx: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """
    Combined unbatchify + gather operation for multi-start results.

    Unbatchifies a flat tensor into [batch, n, ...] and then gathers
    the elements at the specified indices.

    Args:
        x: Flat tensor [batch * n, ...].
        idx: Gather indices [batch].
        n: Number of repeats used in batchify.

    Returns:
        Gathered tensor [batch, ...].
    """
    # X: [batch * n, ...] -> [batch, n, ...]
    bs = x.size(0) // n
    x_reshaped = x.view(bs, n, *x.shape[1:])

    # Idx: [batch] -> [batch, 1, ...]
    # Expand indices to match dims of x_reshaped excluding n
    view_shape = [bs, 1] + [1] * (x.dim() - 1)
    expand_shape = [bs, 1] + list(x.shape[1:])

    gathered_idx = idx.view(*view_shape).expand(*expand_shape)

    # Gather
    out = x_reshaped.gather(1, gathered_idx)

    return out.squeeze(1)
