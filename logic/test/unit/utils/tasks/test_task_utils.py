"""Tests for task utility functions."""

import numpy as np
import torch
from typing import cast
from logic.src.utils.tasks.task_utils import calculate_edges, make_instance_generic


def test_calculate_edges_dist():
    """Verify edge calculation with distance-based strategy."""
    loc = np.array([[0, 0], [0, 1], [1, 0]])
    threshold = 1.1  # Should connect 0-1 and 0-2 but not 1-2 (dist=1.414)

    edges = calculate_edges(loc, threshold, edge_strategy="dist")

    # get_edge_idx_dist returns (2, num_edges)
    assert cast(torch.Tensor, edges).shape[0] == 2
    assert cast(torch.Tensor, edges).shape[1] == 12


def test_calculate_edges_knn():
    """Verify edge calculation with KNN-based strategy."""
    loc = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # k=2 neighbors
    edges = calculate_edges(loc, edge_threshold=2, edge_strategy="knn")

    # Each node should have 2 neighbors + depot connections (if add_depot=True)
    assert cast(torch.Tensor, edges).shape[0] == 2
    assert cast(torch.Tensor, edges).shape[1] == 16


def test_calculate_edges_invalid():
    """Verify handling of invalid threshold or strategy."""
    loc = np.array([[0, 0], [1, 1]])

    # Threshold <= 0
    assert calculate_edges(loc, 0, "dist") is None

    # Unknown strategy
    assert calculate_edges(loc, 1, "unknown") is None


def test_make_instance_generic():
    """Verify creation of problem instances from raw data."""
    depot = [0.5, 0.5]
    loc = [[0, 0], [0, 1], [1, 0]]
    waste = [10, 20, 30]
    max_waste = 100

    args = (depot, loc, waste, max_waste, [])  # add dummy *rest
    instance = make_instance_generic(args, edge_threshold=1.1, edge_strategy="dist")

    assert torch.is_tensor(instance["loc"])
    assert torch.is_tensor(instance["depot"])
    assert torch.is_tensor(instance["waste"])
    assert torch.is_tensor(instance["max_waste"])
    assert "edges" in instance
    assert instance["loc"].shape == (3, 2)
    assert instance["waste"].shape == torch.Size([])


def test_make_instance_generic_multi_day():
    """Verify handling of multi-day waste levels."""
    depot = [0.5, 0.5]
    loc = [[0, 0], [0, 1]]
    # 2 days, 2 nodes
    waste = [[10, 20], [15, 25]]
    max_waste = 100

    args = (depot, loc, waste, max_waste, [])
    instance = make_instance_generic(args, edge_threshold=0, edge_strategy="dist")

    # waste should be first day
    assert torch.equal(instance["waste"], torch.tensor([10.0, 20.0]))
    # fill1 should be second day
    assert torch.equal(instance["fill1"], torch.tensor([15.0, 25.0]))
    assert "edges" not in instance
