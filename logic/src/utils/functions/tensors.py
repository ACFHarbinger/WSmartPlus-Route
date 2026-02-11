"""
Tensor manipulation and device movement utilities.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, TypeVar, cast

import torch

from logic.src.interfaces import ITraversable

T = TypeVar("T")


def move_to(var: T, device: torch.device, non_blocking: bool = False) -> T:
    """
    Recursively moves variables to the specified device.
    Supports dicts and Tensors.

    Args:
        var: The variable to move.
        device: The target device.
        non_blocking: If True and if the variable is a Tensor on pinned memory,
            the copy will be asynchronous with respect to the host. Defaults to False.

    Returns:
        The variable on the new device.
    """
    if var is None:
        return var  # type: ignore
    if hasattr(var, "to") and callable(var.to):
        return var.to(device, non_blocking=non_blocking)  # type: ignore
    if isinstance(var, ITraversable):
        return {k: move_to(v, device, non_blocking=non_blocking) for k, v in var.items()}  # type: ignore
    if isinstance(var, (list, tuple)):
        return type(var)(move_to(v, device, non_blocking=non_blocking) for v in var)  # type: ignore
    return var


def compute_in_batches(
    f: Callable[..., Any],
    calc_batch_size: int,
    *args: torch.Tensor,
    n: Optional[int] = None,
) -> Any:
    """
    Computes memory heavy function f(*args) in batches.

    Args:
        f: The function that is computed, should take only tensors as arguments and
            return tensor or tuple of tensors.
        calc_batch_size: The batch size to use when computing this function.
        *args: Tensor arguments with equally sized first batch dimension.
        n: the total number of elements.

    Returns:
        f(*args), this should be one or multiple tensors.
    """
    if n is None:
        n = args[0].size(0)
    n_batches = (n + calc_batch_size - 1) // calc_batch_size  # ceil
    if n_batches == 1:
        return f(*args)

    # Run all batches
    all_res = [f(*(arg[i * calc_batch_size : (i + 1) * calc_batch_size] for arg in args)) for i in range(n_batches)]

    # Allow for functions that return None
    def safe_cat(chunks: List[Optional[torch.Tensor]], dim: int = 0) -> Optional[torch.Tensor]:
        """Concatenates tensors safely, handling empty chunks."""
        if chunks[0] is None:
            assert all(chunk is None for chunk in chunks)
            return None
        return torch.cat(cast(List[torch.Tensor], chunks), dim)

    # Depending on whether the function returned a tuple we need to concatenate each element or only the result
    if isinstance(all_res[0], tuple):
        return tuple(safe_cat(list(res_chunks), 0) for res_chunks in zip(*all_res))
    return safe_cat(all_res, 0)


def do_batch_rep(v: Any, n: int) -> Any:
    """
    Replicates a variable n times along the batch dimension.

    Args:
        v: The variable (tensor, or structure containing tensors).
        n: Number of repetitions.

    Returns:
        Replicated variable.
    """
    if v is None:
        return None
    if isinstance(v, ITraversable):
        # We need a recursive call that works for nested dicts
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)
    if isinstance(v, torch.Tensor):
        return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])

    # Handle TensorDict or other objects with unsqueeze/expand/reshape
    if hasattr(v, "unsqueeze") and hasattr(v, "batch_size"):
        # TensorDict-like
        return v.unsqueeze(0).expand(n, *v.batch_size).clone().reshape(-1)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])
