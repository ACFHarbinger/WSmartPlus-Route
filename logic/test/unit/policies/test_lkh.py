
import numpy as np
import pytest
from logic.src.policies.lin_kernighan_helsgaun import solve_lkh

def test_lkh_small_tsp():
    """Test LKH on a small 5-node TSP."""
    # 5 nodes in a line: 0-1-2-3-4
    # Distances:
    # 0-1: 10
    # 1-2: 10
    # 2-3: 10
    # 3-4: 10
    # 4-0: 100 (closes the loop far away)
    # Optimal tour: 0-1-2-3-4-0 (Cost 140)
    # Or 0-4-3-2-1-0 (Cost 140)

    # Let's make a grid for easier validation
    # 0(0,0)  1(0,10)
    # 3(10,0) 2(10,10)
    # Optimal: 0-1-2-3-0 cost = 10+10+10+10=40

    coords = np.array([
        [0, 0],
        [0, 10],
        [10, 10],
        [10, 0]
    ])

    n = 4
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    tour, cost = solve_lkh(dist_matrix, max_iterations=50)

    # Check validity
    assert len(tour) == n + 1
    assert tour[0] == tour[-1]
    assert len(set(tour)) == n

    # Check optimality
    assert abs(cost - 40.0) < 1e-5

def test_lkh_candidates():
    """Verify LKH generates valid candidates."""
    # 3 nodes triangle 3-4-5
    # 0-1: 3
    # 1-2: 4
    # 2-0: 5
    d = np.array([
        [0, 3, 5],
        [3, 0, 4],
        [5, 4, 0]
    ])
    tour, cost = solve_lkh(d)
    assert len(tour) == 4
    assert cost == 12.0

def test_apply_3opt_move():
    """Test 3-opt move application directly."""
    # Tour: 0-1-2-3-4-5-0
    tour = [0, 1, 2, 3, 4, 5, 0]

    # i=0 (v0), j=2 (v2), k=4 (v4)
    # Case 0: reverses (0+1..2) i.e. 1-2 -> 2-1
    #         reverses (2+1..4) i.e. 3-4 -> 4-3
    # Expected: 0, 2, 1, 4, 3, 5, 0

    from logic.src.policies.lin_kernighan_helsgaun import apply_3opt_move
    new_tour = apply_3opt_move(tour, 0, 2, 4, case=0)
    expected = [0, 2, 1, 4, 3, 5, 0]
    assert new_tour == expected, f"Expected {expected}, got {new_tour}"
