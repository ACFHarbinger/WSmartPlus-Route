
import numpy as np
import pandas as pd
from collections import deque
from unittest.mock import patch, MagicMock
import sys
import os

# Mock missing dependencies to allow importing solver without triggering the whole tree
from unittest.mock import MagicMock
sys.modules["alns"] = MagicMock()
sys.modules["alns.accept"] = MagicMock()
sys.modules["alns.select"] = MagicMock()
sys.modules["alns.stop"] = MagicMock()
sys.modules["gurobipy"] = MagicMock()
sys.modules["hexaly"] = MagicMock()
sys.modules["pyvrp"] = MagicMock()
sys.modules["vrpy"] = MagicMock()
sys.modules["fast_tsp"] = MagicMock()
sys.modules["googlemaps"] = MagicMock()
sys.modules["geopy"] = MagicMock()
sys.modules["geopandas"] = MagicMock()
sys.modules["osmnx"] = MagicMock()
sys.modules["shapely"] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath("."))

from logic.src.policies.popmusic.solver import run_popmusic, find_route_neighbors
from scipy.spatial import KDTree

def test_find_route_neighbors_kdtree():
    print("Testing KD-Tree neighbor search...")
    centroids = [
        np.array([0, 0]),
        np.array([1, 1]),
        np.array([10, 10]),
        np.array([11, 11])
    ]
    kdtree = KDTree(np.array(centroids))

    # k=2 neighbors of seed 0 should be [0, 1]
    neighbors = find_route_neighbors(0, centroids, k=2, kdtree=kdtree, k_prox=4)
    assert set(neighbors) == {0, 1}, f"Expected {0, 1}, got {set(neighbors)}"

    # k=2 neighbors of seed 2 should be [2, 3]
    neighbors = find_route_neighbors(2, centroids, k=2, kdtree=kdtree, k_prox=4)
    assert set(neighbors) == {2, 3}, f"Expected {2, 3}, got {set(neighbors)}"
    print("KD-Tree neighbor search passed!")

def test_popmusic_lifo_order():
    print("Testing POPMUSIC LIFO order...")
    coords = pd.DataFrame({
        "Lat": [0, 1, 10, 11],
        "Lng": [0, 10, 1, 11]
    })
    dist_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            dist_matrix[i, j] = np.linalg.norm(coords.iloc[i].values - coords.iloc[j].values)

    data = {
        "coords": coords,
        "must_go": [1, 2, 3],
        "distance_matrix": dist_matrix,
        "n_vehicles": 4,
        "wastes": {1: 10, 2: 10, 3: 10},
        "capacity": 100,
        "R": 1.0,
        "C": 0.1
    }

    call_order = []
    def mock_optimize(*args, **kwargs):
        seed_idx = kwargs["neighborhood_indices"][0]
        call_order.append(seed_idx)
        # Improve only on the first call to seed 3 to see if it re-pushes
        if seed_idx == 3 and len(call_order) == 1:
            return [[0, 1, 0]], 100.0  # Big improvement
        return [[0, seed_idx, 0]], 0.0 # No improvement

    with patch("logic.src.policies.popmusic.solver._optimize_subproblem", side_effect=mock_optimize):
        run_popmusic(**data, subproblem_size=2, seed_strategy="lifo")

    print(f"Call order: {call_order}")
    # Initial stack for 4 routes is [0, 1, 2, 3].
    # 1. Pop 3. R=[3, 2]. Improve! Re-push 3, 2. Stack: [0, 1, 3, 2]
    # 2. Pop 2. No improvement. Stack: [0, 1, 3]
    # 3. Pop 3. No improvement. Stack: [0, 1]
    # 4. Pop 1. No improvement. Stack: [0]
    # 5. Pop 0. No improvement. Stack: []
    assert call_order[0] == 3
    assert call_order[1] == 2
    assert call_order[2] == 3
    print("POPMUSIC LIFO order passed!")

if __name__ == "__main__":
    try:
        test_find_route_neighbors_kdtree()
        test_popmusic_lifo_order()
        print("\nALL STANDALONE TESTS PASSED!")
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
