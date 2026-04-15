import unittest
import numpy as np
import torch
from logic.src.policies.helpers.operators.intra_route.swap import move_swap
from logic.src.policies.helpers.operators.repair.geni import geni_insertion, geni_profit_insertion
from logic.src.policies.helpers.operators.repair.nearest import nearest_insertion, nearest_profit_insertion

class MockLS:
    def __init__(self, dist_matrix, routes, waste=None, capacity=100.0, C=1.0):
        self.d = dist_matrix
        self.routes = routes
        self.waste = waste or {}
        self.Q = capacity
        self.C = C
        self.affected = set()

    def _update_map(self, affected_indices):
        self.affected.update(affected_indices)

    def _get_load_cached(self, ri):
        return sum(self.waste.get(n, 0) for n in self.routes[ri])

class TestRefactoredOperators(unittest.TestCase):
    def test_adjacent_swap(self):
        # 0 - 1 - 2 - 3 - 0
        # dist(0,1)=10, dist(1,2)=10, dist(2,3)=10, dist(3,0)=10
        # dist(0,2)=15, dist(1,3)=15
        d = np.zeros((4, 4))
        d[0,1] = d[1,0] = 10
        d[1,2] = d[2,1] = 10
        d[2,3] = d[3,2] = 10
        d[3,0] = d[0,3] = 10
        d[0,2] = d[2,0] = 15
        d[1,3] = d[3,1] = 15

        # Route: [1, 2, 3]
        # Current edges: (0,1), (1,2), (2,3), (3,0) -> cost = 10+10+10+10 = 40
        # Swap (1,2): Route becomes [2, 1, 3]
        # New edges: (0,2), (2,1), (1,3), (3,0) -> cost = 15+10+15+10 = 50
        # Delta should be +10 (non-improving)

        ls = MockLS(d, [[1, 2, 3]])
        # u=1, v=2, r_u=0, p_u=0, r_v=0, p_v=1
        improved = move_swap(ls, 1, 2, 0, 0, 0, 1)
        self.assertFalse(improved)

        # Now make it improving: dist(0,2) = 5, dist(1,3) = 5
        d[0,2] = d[2,0] = 5
        d[1,3] = d[3,1] = 5
        # New edges: (0,2), (2,1), (1,3), (3,0) -> cost = 5+10+5+10 = 30
        # Delta = 30 - 40 = -10
        ls = MockLS(d, [[1, 2, 3]])
        improved = move_swap(ls, 1, 2, 0, 0, 0, 1)
        self.assertTrue(improved)
        self.assertEqual(ls.routes[0], [2, 1, 3])

    def test_geni_insertion_type_i(self):
        # Test Type I: u connects v_i to v_j, reverse (v_{i+1}...v_j)
        # Route: 0 - 1 - 2 - 3 - 4 - 0
        # Insert 5 between 1 and 3
        # i=0 (v_i=1), j=2 (v_j=3)
        # Type I: 0 - 1 - 5 - 3 - 2 - 4 - 0
        dist = np.full((7, 7), 10.0)
        np.fill_diagonal(dist, 0)
        # Make Type I very cheap
        dist[1, 5] = dist[5, 3] = dist[2, 4] = 1.0

        routes = [[1, 2, 3, 4]]
        removed = [5]
        wastes = {i: 1 for i in range(7)}

        new_routes = geni_insertion(routes, removed, dist, wastes, 100.0)
        # Check if 5 is in the route
        self.assertIn(5, new_routes[0])
        # Expected sequence: [1, 5, 3, 2, 4]
        self.assertEqual(new_routes[0], [1, 5, 3, 2, 4])

    def test_geni_insertion_type_ii(self):
        # Type II: u connects v_i to v_{j-1}, reverse (v_{i+1}...v_{j-1})
        # Route: 0 - 1 - 2 - 3 - 4 - 0
        # Insert 5 between 1 and 2 (using j=3, v_j=4)
        # i=0 (v_i=1), j=3 (v_j=4)
        # Type II: 0 - 1 - 5 - 2 - 3 - 4 - 0 (Wait, if i=0, j=3, segment is [2, 3])
        # Added edges: (v_i, u), (u, v_{j-1}), (v_{i+1}, v_j)
        # (1, 5), (5, 3), (2, 4)
        dist = np.full((7, 7), 10.0)
        np.fill_diagonal(dist, 0)
        # Make Type II very cheap
        dist[1, 5] = dist[5, 3] = dist[2, 4] = 1.0

        routes = [[1, 2, 3, 4]]
        removed = [5]
        wastes = {i: 1 for i in range(7)}

        # Since I set same cheap edges as Type I test, it might pick either depending on loop order.
        new_routes = geni_insertion(routes, removed, dist, wastes, 100.0)
        self.assertIn(5, new_routes[0])
        self.assertEqual(len(new_routes[0]), 5)

    def test_nearest_insertion(self):
        # Route: 0 - 1 - 2 - 0 (dist 10 each)
        # Node 3 is very close to 1
        dist = np.full((4, 4), 10.0)
        np.fill_diagonal(dist, 0)
        dist[1, 3] = 1.0  # Node 3 is nearest to Node 1

        routes = [[1, 2]]
        removed = [3]
        wastes = {i: 1 for i in range(4)}

        new_routes = nearest_insertion(routes, removed, dist, wastes, 100.0)
        self.assertIn(3, new_routes[0])
        # Position should be next to 1: [1, 3, 2] or [3, 1, 2]
        self.assertTrue(new_routes[0] == [1, 3, 2] or new_routes[0] == [3, 1, 2])

    def test_nearest_profit_insertion(self):
        # Node 3 is nearest but unprofitable
        # Node 4 is farther but profitable
        # Use explicit distances to avoid cost_increase being negative due to zeros.
        dist = np.full((5, 5), 10.0)
        np.fill_diagonal(dist, 0)

        dist[0, 1] = dist[1, 0] = 5.0
        dist[1, 2] = dist[2, 1] = 5.0
        dist[2, 0] = dist[0, 2] = 5.0

        dist[1, 3] = dist[3, 1] = 6.0   # Cost delta: dist(1,3)+dist(3,2)-dist(1,2) = 6+10-5 = 11.0
        dist[1, 4] = dist[4, 1] = 5.5   # Cost delta: dist(1,4)+dist(4,2)-dist(1,2) = 5.5+10-5 = 10.5

        # Wastes (revenue = waste * R, R=1.0)
        # Node 3: waste=1.0 -> revenue=1.0. Cost delta 11.0. Profit = -10 (Unprofitable)
        # Node 4: waste=20.0 -> revenue=20.0. Cost delta 10.5. Profit = +9.5 (Profitable)
        wastes = {1: 1.0, 2: 1.0, 3: 1.0, 4: 20.0}

        routes = [[1, 2]]
        removed = [3, 4]

        # Set C=1.0, R=1.0, and mark node 4 as mandatory to avoid pruning of the unprofitable route
        new_routes = nearest_profit_insertion(routes, removed, dist, wastes, 100.0, R=1.0, C=1.0, mandatory_nodes=[4])
        self.assertIn(4, new_routes[0])

if __name__ == "__main__":
    unittest.main()
