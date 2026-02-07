import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from logic.src.pipeline.simulations.network import (
    EuclideanStrategy,
    GeodesicStrategy,
    HaversineStrategy,
    apply_edges,
    compute_distance_matrix,
    get_paths_between_states,
)


class TestNetwork:
    @pytest.fixture
    def coords(self):
        return pd.DataFrame({"ID": [1, 2, 3], "Lat": [0.0, 0.0, 1.0], "Lng": [0.0, 1.0, 0.0]})

    def test_haversine_strategy(self, coords):
        strategy = HaversineStrategy()
        dm = strategy.calculate(coords)
        assert dm.shape == (3, 3)
        assert dm[0, 0] == 0.0
        # Distance between (0,0) and (0,1) deg ~ 111km
        assert dm[0, 1] > 100
        assert dm[0, 1] < 120

    def test_euclidean_strategy(self, coords):
        strategy = EuclideanStrategy()
        dm = strategy.calculate(coords)
        assert dm.shape == (3, 3)
        assert dm[0, 0] == 0.0
        # Euclidean logic: 86.51 * 1.58 * sqrt(dlat^2 + dlng^2)
        # (0,0) to (0,1) -> dist = 1 -> ~136.68
        assert dm[0, 1] > 100

    def test_geodesic_strategy(self, coords):
        strategy = GeodesicStrategy()
        # Geodesic might be slow, test small
        dm = strategy.calculate(coords)
        assert dm.shape == (3, 3)

    def test_compute_distance_matrix_caching(self, coords, tmp_path):
        dm_file = tmp_path / "test_dm.csv"

        # 1. Compute and save
        with patch("logic.src.pipeline.simulations.network.ROOT_DIR", str(tmp_path)):
            # Mock os.path.join to just use tmp_path
            # But compute_distance_matrix uses ROOT_DIR/data/...
            # Better to just mock open or use relative path?
            # It uses logic: if filename_only -> join ROOT_DIR...
            # if absolute config -> use as is.

            # Using absolute path for dm_filepath
            str_path = str(dm_file)
            compute_distance_matrix(coords, "hsd", dm_filepath=str_path)

            assert os.path.exists(dm_file)

            # 2. Load from cache
            # Mod coords to ensure we are loading old data (if we were recalculating, it would change)
            # But here we just check if it reads the file.
            # Let's mock loadtxt to be sure
            with patch("numpy.loadtxt") as mock_load:
                mock_load.return_value = np.zeros((4, 4))  # Return dummy including header col/row logic

                compute_distance_matrix(coords, "hsd", dm_filepath=str_path)
                mock_load.assert_called_once()

    def test_apply_edges_dist(self):
        # 4 Nodes: 0 (Depot), 1, 2, 3
        # 0-1: 10
        # 1-2: 20 (Target long edge)
        # 2-3: 10 (Target short edge)
        # 0-2: 30 (Long depot edge)

        # Ensure float to avoid OverflowError
        dist_matrix = np.full((4, 4), 100.0)  # Init with large values
        np.fill_diagonal(dist_matrix, 0.0)

        # Set specific edges
        dist_matrix[0, 1] = 10.0
        dist_matrix[1, 0] = 10.0
        dist_matrix[1, 2] = 20.0
        dist_matrix[2, 1] = 20.0
        dist_matrix[2, 3] = 10.0
        dist_matrix[3, 2] = 10.0
        dist_matrix[0, 2] = 30.0
        dist_matrix[2, 0] = 30.0

        # Threshold: Keep top 1 edge (Global sparsification by distance).
        # Edges (clients only):
        # (2,3) dist 10
        # (1,2) dist 20
        # (1,3) dist 100
        # If we keep 1, only (2,3) remains. (1,2) is removed.

        dm_edges, paths, adj = apply_edges(dist_matrix, edge_thresh=1, edge_method="dist")

        # Verify adjacency
        assert adj[1, 2] == 0  # 20 > 10, not in top 1
        assert adj[2, 3] == 1  # 10 is shortest, kept
        assert adj[0, 2] == 1  # Depot preserved

        # Shortest path 1->2.
        # Direct 1-2 removed (was 20).
        # Path 1 -> 0 -> 2 (10 + 30 = 40).
        # Or 1 -> 0 -> 1 ...
        # (1,2) should not have direct edge (which was 20).
        # FW keeps weights where adj=1.
        # dm_edges[1,2] should be path length via kept edges.
        # Path: 1 -> 0 -> 2 (10 + 30 = 40).
        # Original direct was 20.
        # So distance increases.

        if dm_edges[1, 2] != float("inf"):
            assert dm_edges[1, 2] > 20.0

    def test_apply_edges_no_sparsification(self):
        dist_matrix = np.ones((2, 2))
        dm, paths, adj = apply_edges(dist_matrix, edge_thresh=0, edge_method="dist")
        assert adj is None
        assert paths is None
        assert np.allclose(dm, dist_matrix)

    def test_get_paths_between_states(self):
        # Must provide valid paths for all pairs (excluding diagonal if implementation logic dictates)
        # Implementation loops ii in 0..n, jj in 0..n.
        # If ii!=jj, access shortest_paths[(ii,jj)]
        paths_dict = {(0, 1): [0, 1], (1, 0): [1, 0]}
        res = get_paths_between_states(2, paths_dict)
        assert res[0][1] == [0, 1]
        assert res[0][0] == [0, 0]  # Default loop
