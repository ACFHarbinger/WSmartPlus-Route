"""
Shared utilities for decoding.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Filter logits to keep only top-k values.

    Args:
        logits: Logits tensor [batch, num_nodes]
        k: Number of top values to keep

    Returns:
        Filtered logits with others set to -inf
    """
    if k <= 0:
        return logits

    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k, dim=-1)
    min_value = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_value, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.

    Keep the smallest set of logits whose cumulative probability >= p.

    Args:
        logits: Logits tensor [batch, num_nodes]
        p: Cumulative probability threshold

    Returns:
        Filtered logits with others set to -inf
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    # Scatter back to original ordering
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def batchify(td: TensorDict, num_repeats: int) -> TensorDict:
    """
    Repeat TensorDict for multistart.

    Args:
        td: TensorDict [batch, ...]
        num_repeats: Number of times to repeat

    Returns:
        TensorDict [batch * num_repeats, ...]
    """
    # Use interleave to keep instances together
    # Some TensorDict versions might not have repeat_interleave on the object
    try:
        return td.repeat_interleave(num_repeats, dim=0)
    except AttributeError:
        # Fallback: repeat manually for each key
        # We must set batch_size FIRST to allow item assignment of different shape
        new_batch_size = torch.Size([td.batch_size[0] * num_repeats, *td.batch_size[1:]])
        new_td = TensorDict({}, batch_size=new_batch_size, device=td.device)
        for key, val in td.items():
            if isinstance(val, torch.Tensor):
                new_td[key] = val.repeat_interleave(num_repeats, dim=0)
        return new_td


def unbatchify(td: TensorDict | torch.Tensor, num_repeats: int) -> TensorDict | torch.Tensor:
    """
    Unbatchify a TensorDict/Tensor by extracting the original batch.
    Args:
        td: TensorDict/Tensor [batch * num_repeats, ...]
        num_repeats: Number of repeats
    Returns:
        Unbatchified object. If Tensor, [batch, num_repeats, ...].
        If TensorDict, [batch, ...] (takes first repeat).
    """
    if isinstance(td, torch.Tensor):
        return td.view(-1, num_repeats, *td.shape[1:])

    batch_size = td.batch_size[0] // num_repeats
    indices = torch.arange(0, batch_size * num_repeats, num_repeats, device=td.device)
    return td[indices]


def gather_by_index(src: torch.Tensor, idx: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Gather elements from src using indices.

    Args:
        src: Source tensor
        idx: Index tensor
        dim: Dimension to gather along

    Returns:
        Gathered tensor
    """
    idx = idx.unsqueeze(-1).expand(*idx.shape, src.shape[-1])
    return src.gather(dim, idx)


def get_log_likelihood(
    log_probs: torch.Tensor,
    actions: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    return_sum: bool = True,
) -> torch.Tensor:
    """
    Get log likelihood of selected actions.

    Args:
        log_probs: Log probabilities [batch, seq_len, num_nodes] or [batch, num_nodes].
        actions: Action indices [batch, seq_len] or [batch]. If None, assumes log_probs are already selected.
        mask: Optional mask to select only certain actions [batch, seq_len].
        return_sum: Whether to sum log probs (True) or return sequence (False).

    Returns:
        Log likelihood tensor. Shape [batch] if return_sum=True, else [batch, seq_len].
    """
    if actions is None:
        # Assume log_probs are already the selected log probabilities
        log_ll = log_probs
    else:
        # Gather log probs for selected actions
        if log_probs.dim() == 3:
            # [batch, seq_len, num_nodes] -> gather on last dim
            log_ll = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        else:
            # [batch, num_nodes] -> gather on last dim
            log_ll = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Apply mask if provided
    if mask is not None:
        log_ll = log_ll * mask

    # Return sum or sequence
    if return_sum:
        return log_ll.sum(dim=-1) if log_ll.dim() > 1 else log_ll
    return log_ll


def modify_logits_for_top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Filter logits to keep only top-k values. Alias for top_k_filter for rl4co compatibility.

    Args:
        logits: Logits tensor [batch, num_nodes]
        top_k: Number of top values to keep

    Returns:
        Filtered logits with others set to -inf
    """
    return top_k_filter(logits, top_k)


def modify_logits_for_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling. Alias for top_p_filter for rl4co compatibility.

    Args:
        logits: Logits tensor [batch, num_nodes]
        top_p: Cumulative probability threshold

    Returns:
        Filtered logits with others set to -inf
    """
    return top_p_filter(logits, top_p)
