"""Unit tests for graph_utils.py."""

import numpy as np
import torch
import pytest
from unittest.mock import MagicMock
import networkx as nx

from logic.src.utils.functions import graph_utils
from logic.src.utils.functions.graph_utils import (
    generate_adj_matrix,
    get_edge_idx_dist,
    sort_by_pairs,
    get_adj_knn,
    adj_to_idx,
    idx_to_adj,
    tour_to_adj,
    get_adj_osm,
    find_longest_path
)

def test_generate_adj_matrix_random():
    """Test random adjacency matrix generation."""
    size = 5
    num_edges = 6
    adj = generate_adj_matrix(size, num_edges, undirected=False, add_depot=False)
    assert adj.shape == (5, 5)
    assert np.sum(adj) == 6
    assert np.all(np.diag(adj) == 0)

    # Undirected
    adj_u = generate_adj_matrix(size, num_edges, undirected=True, add_depot=False)
    assert np.sum(adj_u) == 12 # 6 edges * 2 directions
    assert np.all(adj_u == adj_u.T)

def test_generate_adj_matrix_with_depot():
    """Test adjacency matrix generation with depot padding."""
    size = 4
    num_edges = 2
    adj = generate_adj_matrix(size, num_edges, undirected=False, add_depot=True)
    assert adj.shape == (5, 5)
    # Depot (node 0) should be connected to everyone (pad with constant_values=1)
    # Then diagonal is filled with 0.
    assert np.all(adj[0, 1:] == 1)
    assert np.all(adj[1:, 0] == 1)
    assert adj[0, 0] == 0

def test_get_edge_idx_dist_symmetric():
    """Test edge index generation from distance matrix (symmetric/undirected)."""
    dist = np.array([
        [0, 1, 5],
        [1, 0, 2],
        [5, 2, 0]
    ], dtype=float)
    # size=3. undirected=True. k=1 edge.
    # Sorted upper tri: (0,1)=1, (1,2)=2, (0,2)=5
    # top 1 edge is (0,1)
    res = get_edge_idx_dist(dist, num_edges=1, add_depot=False, undirected=True)
    # res is [2, 1] -> [[0], [1]]
    assert res.shape == (2, 1)
    assert np.array_equal(res, [[0], [1]])

def test_sort_by_pairs():
    """Test sorting edge indices by linear node index."""
    edge_idx = torch.tensor([[1, 0, 2], [2, 1, 0]]) # (1,2), (0,1), (2,0)
    # sorted order based on i*N + j: (0,1), (1,2), (2,0)
    sorted_idx = sort_by_pairs(3, edge_idx)
    expected = torch.tensor([[0, 1, 2], [1, 2, 0]])
    assert torch.equal(sorted_idx, expected)

def test_get_adj_knn():
    """Test KNN-based adjacency matrix generation."""
    dist = np.array([
        [0, 1, 10, 10],
        [1, 0, 10, 10],
        [10, 10, 0, 1],
        [10, 10, 1, 0]
    ], dtype=float)
    # size=4, k=1. Each node should have closest neighbor.
    # Node 0: NN is 1. Node 1: NN is 0. Node 2: NN is 3. Node 3: NN is 2.
    # negative=False (1=edge)
    adj = get_adj_knn(dist, k_neighbors=1, add_depot=False, negative=False)
    assert adj[0, 1] == 1
    assert adj[0, 2] == 0
    assert adj[2, 3] == 1
    assert np.all(np.diag(adj) == 0)

def test_adj_idx_conversion():
    """Test conversion between adjacency matrix and edge indices."""
    adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 0]
    ])
    # negative=False (1 means edge)
    idx = adj_to_idx(adj, negative=False)
    # idx: [[0, 1, 1], [1, 0, 2]]
    assert idx.shape == (2, 3)

    adj_reconstructed = idx_to_adj(torch.from_numpy(idx), negative=False)
    assert np.array_equal(adj, adj_reconstructed)

def test_tour_to_adj():
    """Test tour sequence to adjacency matrix."""
    tour = [0, 2, 1]
    # edges: (0,2), (2,1), (1,0) - undirected
    adj = tour_to_adj(tour)
    expected = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    assert np.array_equal(adj, expected)

def test_find_longest_path():
    """Test finding the longest path in a small graph."""
    # 0 -> 1 (w=5)
    # 1 -> 2 (w=10)
    # 0 -> 2 (w=2)
    dist = torch_full = torch.full((3, 3), float("-inf"))
    dist[0, 1] = 5.0
    dist[1, 2] = 10.0
    dist[0, 2] = 2.0

    length, path = find_longest_path(dist, start_vertex=0)
    # Longest path: 0 -> 1 -> 2 (length 15)
    assert length == 15.0
    assert path == [0, 1, 2]

def test_generate_adj_matrix_percentage():
    """Test generating adjacency matrix using percentage of edges."""
    size = 10
    # num_edges as float (percentage)
    # total possible directed = 10 * 9 = 90. 10% = 9 edges.
    adj = generate_adj_matrix(size, 0.1, undirected=False, add_depot=False)
    assert np.sum(adj) == 9

def test_get_edge_idx_dist_percentage():
    """Test generating edge indices from distance matrix using percentage."""
    dist = np.random.rand(10, 10)
    dist = (dist + dist.T) / 2 # symmetric
    np.fill_diagonal(dist, 0)
    # total possible undirected = 10 * 9 / 2 = 45. 20% = 9 edges.
    idx = get_edge_idx_dist(dist, 0.2, add_depot=False, undirected=True)
    assert idx.shape[1] == 9

def test_find_longest_path_cycle():
    """Test longest path search in a cyclic graph."""
    # 0 -> 1 (5), 1 -> 2 (10), 2 -> 0 (100)
    # If we return to start, length is 5+10+100=115
    dist = torch.full((3, 3), float("-inf"))
    dist[0, 1] = 5.0
    dist[1, 2] = 10.0
    dist[2, 0] = 100.0

    length, path = find_longest_path(dist, start_vertex=0)
    assert length == 115.0
    assert path == [0, 1, 2, 0]

def test_get_adj_osm_full(mocker):
    """Test get_adj_osm with full mocking of osmnx and networkx."""
    mocker.patch("osmnx.distance.nearest_nodes", return_value=123)
    mocker.patch("networkx.to_numpy_array", return_value=np.zeros((2, 2)))

    G = nx.MultiDiGraph()
    coords = MagicMock()
    coords.shape = (2, 2)
    coords.copy.return_value = coords

    adj = get_adj_osm(coords, 2, [G], add_depot=False, negative=False)
    assert adj.shape == (2, 2)
