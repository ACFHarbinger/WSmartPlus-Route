"""
Unified tensor operations for combinatorial optimization.

This module provides commonly-used tensor operations for VRP/CO problems,
consolidating scattered utilities into a single discoverable location.

Operations include:
- Distance and tour length computation
- Graph sparsification and edge index construction
- Multi-start action selection and gathering
- Coordinate transformations
- Batch scatter operations

Reference: Adapted from rl4co/utils/ops.py patterns, tailored for WSmart-Route.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch
from tensordict import TensorDict

# Re-export existing operations from decoding.py for convenience
from logic.src.utils.functions.decoding import batchify, gather_by_index, unbatchify

# =============================================================================
# Distance and Tour Operations
# =============================================================================


def get_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean distance between two batched coordinate tensors.

    Args:
        x: Coordinates of shape [..., 2] (or [..., d] for d-dimensional).
        y: Coordinates of shape [..., 2] (or [..., d] for d-dimensional).

    Returns:
        Distance tensor of shape [...].
    """
    return (x - y).norm(p=2, dim=-1)


def get_distance_matrix(locs: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distance matrix from coordinates.

    Args:
        locs: Node coordinates [batch, n_nodes, 2].

    Returns:
        Distance matrix [batch, n_nodes, n_nodes].
    """
    # (x_i - x_j)^2 = x_i^2 + x_j^2 - 2*x_i*x_j
    diff = locs[:, :, None, :] - locs[:, None, :, :]
    return diff.norm(p=2, dim=-1)


def get_tour_length(ordered_locs: torch.Tensor) -> torch.Tensor:
    """
    Compute total tour length for a batch of ordered location sequences.

    Assumes the tour returns to the starting point (closed tour).

    Args:
        ordered_locs: Ordered coordinates [batch, seq_len, 2].

    Returns:
        Tour length [batch].
    """
    # Distances between consecutive nodes
    rolled = torch.roll(ordered_locs, -1, dims=1)
    segment_lengths = get_distance(ordered_locs, rolled)
    return segment_lengths.sum(-1)


def get_open_tour_length(ordered_locs: torch.Tensor) -> torch.Tensor:
    """
    Compute total tour length for open tours (no return to start).

    Args:
        ordered_locs: Ordered coordinates [batch, seq_len, 2].

    Returns:
        Tour length [batch].
    """
    segment_lengths = get_distance(ordered_locs[:, :-1], ordered_locs[:, 1:])
    return segment_lengths.sum(-1)


# =============================================================================
# Entropy and Probability Operations
# =============================================================================


def calculate_entropy(logprobs: torch.Tensor) -> torch.Tensor:
    """
    Calculate entropy from log probabilities.

    H = -sum(p * log(p))

    Args:
        logprobs: Log probabilities [batch, ..., n_actions].

    Returns:
        Entropy [batch, ...].
    """
    probs = logprobs.exp()
    # Mask zeros to avoid nan from 0 * log(0)
    p_log_p = probs * logprobs
    p_log_p = torch.where(probs > 0, p_log_p, torch.zeros_like(p_log_p))
    return -p_log_p.sum(dim=-1)


# =============================================================================
# Multi-Start Node Selection
# =============================================================================


def select_start_nodes(
    td: TensorDict,
    num_starts: int,
) -> torch.Tensor:
    """
    Select start nodes for POMO-style multi-start decoding.

    Selects `num_starts` starting nodes uniformly. If num_starts <= num_nodes,
    uses the first `num_starts` nodes (deterministic for reproducibility).
    Otherwise, samples with replacement.

    Args:
        td: TensorDict with 'locs' key [batch, n_nodes, 2].
        num_starts: Number of start nodes to select.

    Returns:
        Selected start node indices [batch * num_starts].
    """
    num_nodes = td["locs"].shape[1]
    batch_size = td["locs"].shape[0]

    if num_starts <= num_nodes:
        # Deterministic: use nodes 0..num_starts-1 for each batch element
        starts = torch.arange(num_starts, device=td.device).repeat(batch_size)
    else:
        # Random with replacement
        starts = torch.randint(0, num_nodes, (batch_size * num_starts,), device=td.device)

    return starts


def select_start_nodes_by_distance(
    td: TensorDict,
    num_starts: int,
) -> torch.Tensor:
    """
    Select start nodes based on distance from depot.

    Selects nodes farthest from depot as starting points, which can
    improve exploration diversity.

    Args:
        td: TensorDict with 'locs' [batch, n_nodes, 2] and 'depot' [batch, 2].
        num_starts: Number of start nodes to select.

    Returns:
        Selected start node indices [batch * num_starts].
    """
    depot = td.get("depot", td["locs"][:, 0:1])
    if depot.dim() == 2:
        depot = depot.unsqueeze(1)

    # Distance from depot to all nodes
    dists = get_distance(td["locs"], depot)  # [batch, n_nodes]

    # Select top-k farthest nodes
    num_nodes = td["locs"].shape[1]
    k = min(num_starts, num_nodes)
    _, top_indices = dists.topk(k, dim=-1)  # [batch, k]

    if num_starts <= num_nodes:
        return top_indices.reshape(-1)
    else:
        # Need more starts than nodes, repeat with some randomness
        batch_size = td["locs"].shape[0]
        extra = num_starts - k
        extra_indices = torch.randint(0, num_nodes, (batch_size, extra), device=td.device)
        all_indices = torch.cat([top_indices, extra_indices], dim=-1)
        return all_indices.reshape(-1)


def get_num_starts(td: TensorDict, env_name: Optional[str] = None) -> int:
    """
    Get environment-specific number of start nodes for multi-start decoding.

    For POMO, typically num_starts = num_nodes (excluding depot).

    Args:
        td: TensorDict with problem data.
        env_name: Optional environment name for special cases.

    Returns:
        Number of start nodes.
    """
    num_nodes = td["locs"].shape[1]
    # For depot-based problems, exclude depot
    if "depot" in td.keys():
        return num_nodes - 1
    return num_nodes


def get_best_actions(
    actions: torch.Tensor,
    max_idxs: torch.Tensor,
) -> torch.Tensor:
    """
    Select best actions from multi-start rollouts.

    Args:
        actions: Actions from all starts [batch * num_starts, seq_len].
        max_idxs: Indices of best starts per batch element [batch].

    Returns:
        Best actions [batch, seq_len].
    """
    num_starts = actions.shape[0] // max_idxs.shape[0]
    # Convert batch indices to flat indices
    flat_idx = max_idxs + torch.arange(0, max_idxs.shape[0], device=max_idxs.device) * num_starts
    return actions[flat_idx]


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
    # Unbatchify: [batch*n, ...] -> [batch, n, ...]
    x_unbatched = x.view(-1, n, *x.shape[1:])
    # Gather: select idx-th element from dim 1
    idx_expanded = idx.unsqueeze(-1)
    for _ in range(len(x.shape) - 1):
        idx_expanded = idx_expanded.unsqueeze(-1)
    idx_expanded = idx_expanded.expand(-1, 1, *x.shape[1:])
    return x_unbatched.gather(1, idx_expanded).squeeze(1)


# =============================================================================
# Graph Operations
# =============================================================================


def sparsify_graph(
    cost_matrix: torch.Tensor,
    k_sparse: int,
    self_loop: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a k-nearest-neighbor sparse graph from a cost/distance matrix.

    Args:
        cost_matrix: Distance/cost matrix [n, n] (single instance, not batched).
        k_sparse: Number of nearest neighbors to keep per node.
        self_loop: Whether to include self-loops.

    Returns:
        Tuple of (edge_index [2, num_edges], edge_attr [num_edges, 1]).
    """
    n = cost_matrix.shape[0]
    k = min(k_sparse, n - (0 if self_loop else 1))

    if not self_loop:
        # Set diagonal to large value so self-loops are excluded from top-k
        cost_matrix = cost_matrix.clone()
        cost_matrix.fill_diagonal_(float("inf"))

    # Get k nearest neighbors per node
    _, topk_indices = cost_matrix.topk(k, dim=-1, largest=False)  # [n, k]

    # Build edge_index
    src = torch.arange(n, device=cost_matrix.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = topk_indices.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)  # [2, n*k]

    # Edge attributes (distances)
    edge_attr = cost_matrix[src, dst].unsqueeze(-1)  # [n*k, 1]

    return edge_index, edge_attr


@lru_cache(maxsize=32)
def _cached_full_graph_edge_index(num_node: int, self_loop: bool) -> torch.Tensor:
    """Cached computation of full graph edge index (on CPU)."""
    adj = torch.ones(num_node, num_node)
    if not self_loop:
        adj.fill_diagonal_(0)
    return torch.nonzero(adj, as_tuple=False).t()


def get_full_graph_edge_index(
    num_node: int,
    self_loop: bool = False,
) -> torch.Tensor:
    """
    Get edge index for a complete graph (cached).

    Args:
        num_node: Number of nodes.
        self_loop: Whether to include self-loops.

    Returns:
        Edge index [2, num_edges] on CPU. Move to device as needed.
    """
    return _cached_full_graph_edge_index(num_node, self_loop).clone()


def adj_to_pyg_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """
    Convert adjacency matrix to PyG-style edge_index.

    Args:
        adj: Adjacency matrix [n, n] (binary or weighted).

    Returns:
        Edge index [2, num_edges].
    """
    return torch.nonzero(adj, as_tuple=False).t().contiguous()


# =============================================================================
# Random Action Sampling
# =============================================================================


def sample_n_random_actions(
    td: TensorDict,
    n: int,
) -> torch.Tensor:
    """
    Sample n random valid actions from the action mask.

    Args:
        td: TensorDict with 'action_mask' key [batch, n_actions].
        n: Number of random actions to sample per batch element.

    Returns:
        Sampled action indices [batch, n].
    """
    mask = td["action_mask"]  # [batch, n_actions]

    # Convert mask to probabilities (uniform over valid actions)
    probs = mask.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Sample n actions
    return torch.multinomial(probs, n, replacement=True)


# =============================================================================
# Coordinate Transformations
# =============================================================================


def cartesian_to_polar(
    cartesian: torch.Tensor,
    origin: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert Cartesian coordinates to polar coordinates.

    Args:
        cartesian: Cartesian coordinates [..., 2] (x, y).
        origin: Origin point [..., 2]. If None, uses (0, 0).

    Returns:
        Polar coordinates [..., 2] (r, theta).
    """
    if origin is not None:
        cartesian = cartesian - origin
    x, y = cartesian[..., 0], cartesian[..., 1]
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return torch.stack([r, theta], dim=-1)


# =============================================================================
# Scatter Operations
# =============================================================================


def batched_scatter_sum(
    src: torch.Tensor,
    idx: torch.Tensor,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Batched scatter and sum operation.

    For each batch element, sums values in src that share the same index.

    Args:
        src: Source values [batch, n, ...].
        idx: Index tensor [batch, n] mapping each element to a target bin.
        dim_size: Output size along the scatter dimension. If None, uses max(idx)+1.

    Returns:
        Scattered sums [batch, dim_size, ...].
    """
    if dim_size is None:
        dim_size = int(idx.max().item()) + 1

    batch_size = src.shape[0]
    out_shape = [batch_size, dim_size] + list(src.shape[2:])
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    # Expand idx to match src dimensions
    idx_expanded = idx.unsqueeze(-1).expand_as(src) if src.dim() > 2 else idx

    out.scatter_add_(1, idx_expanded, src)
    return out


# =============================================================================
# Module exports
# =============================================================================


__all__ = [
    # Re-exports
    "batchify",
    "unbatchify",
    "gather_by_index",
    # Distance and tour
    "get_distance",
    "get_distance_matrix",
    "get_tour_length",
    "get_open_tour_length",
    # Entropy
    "calculate_entropy",
    # Multi-start
    "select_start_nodes",
    "select_start_nodes_by_distance",
    "get_num_starts",
    "get_best_actions",
    "unbatchify_and_gather",
    # Graph operations
    "sparsify_graph",
    "get_full_graph_edge_index",
    "adj_to_pyg_edge_index",
    # Random actions
    "sample_n_random_actions",
    # Coordinate transforms
    "cartesian_to_polar",
    # Scatter
    "batched_scatter_sum",
]
