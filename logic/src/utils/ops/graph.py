"""
Graph Operations (Sparsification, Edge Indices).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch


def sparsify_graph(
    cost_matrix: torch.Tensor,
    k_sparse: int,
    self_loop: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a k-nearest-neighbor sparse graph from a cost/distance matrix.

    Args:
        cost_matrix: Distance/cost matrix [n, n] (single instance, not batched).
        k_sparse: Number of nearest neighbors to keep per node.
        self_loop: Whether to include self-loops.

    Returns:
        Tuple of (edge_index [2, num_edges], edge_attr [num_edges, 1]).
    """
    n = cost_matrix.size(0)

    # Ensure k_sparse is valid
    k = min(k_sparse, n - 1)

    # Get top-k smallest values (nearest neighbors)
    # topk returns (values, indices)
    # We want smallest distance -> largest negative distance
    topk_vals, topk_ind = torch.topk(cost_matrix * -1.0, k=k, dim=1, largest=True)

    # Build edge index
    # Source nodes: [0, 0, ..., 1, 1, ...]
    node_idx = torch.arange(n, device=cost_matrix.device).unsqueeze(1).expand(n, k)

    # Convert to flattened edge format [2, num_edges]
    src = node_idx.flatten()
    dst = topk_ind.flatten()
    edge_index = torch.stack([src, dst], dim=0)

    # Edge attributes (distances)
    # Restore positive values
    edge_attr = topk_vals.flatten() * -1.0
    edge_attr = edge_attr.unsqueeze(1)  # [num_edges, 1]

    if self_loop:
        # Add self-loops
        loop_node_idx = torch.arange(n, device=cost_matrix.device)
        loop_edge_index = torch.stack([loop_node_idx, loop_node_idx], dim=0)
        loop_edge_attr = torch.zeros(n, 1, device=cost_matrix.device)

        edge_index = torch.cat([edge_index, loop_edge_index], dim=1)
        edge_attr = torch.cat([edge_attr, loop_edge_attr], dim=0)

    return edge_index, edge_attr


@lru_cache(maxsize=16)
def _cached_full_graph_edge_index(num_node: int, self_loop: bool) -> torch.Tensor:
    """Cached computation of full graph edge index (on CPU)."""
    adj_matrix = torch.ones(num_node, num_node)
    if not self_loop:
        adj_matrix.fill_diagonal_(0)
    return adj_matrix.nonzero().t()


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
    return _cached_full_graph_edge_index(num_node, self_loop)
