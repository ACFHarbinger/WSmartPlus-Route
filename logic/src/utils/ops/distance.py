"""
Distance and Tour Length Operations.
"""

from __future__ import annotations

import torch


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
    # (batch, n, 1, 2) - (batch, 1, n, 2) -> (batch, n, n, 2)
    return (locs.unsqueeze(2) - locs.unsqueeze(1)).norm(p=2, dim=-1)


def get_tour_length(ordered_locs: torch.Tensor) -> torch.Tensor:
    """
    Compute total tour length for a batch of ordered location sequences.

    Assumes the tour returns to the starting point (closed tour).

    Args:
        ordered_locs: Ordered coordinates [batch, seq_len, 2].

    Returns:
        Tour length [batch].
    """
    # ordered_locs: [batch, seq, 2]
    # Shift by 1 to compute distance between i and i+1
    rolled = ordered_locs.roll(shifts=-1, dims=1)
    return (ordered_locs - rolled).norm(p=2, dim=-1).sum(1)


def get_open_tour_length(ordered_locs: torch.Tensor) -> torch.Tensor:
    """
    Compute total tour length for open tours (no return to start).

    Args:
        ordered_locs: Ordered coordinates [batch, seq_len, 2].

    Returns:
        Tour length [batch].
    """
    # Exclude the connection from last to first
    # Distance from i to i+1
    d = (ordered_locs[:, :-1] - ordered_locs[:, 1:]).norm(p=2, dim=-1)
    return d.sum(1)
