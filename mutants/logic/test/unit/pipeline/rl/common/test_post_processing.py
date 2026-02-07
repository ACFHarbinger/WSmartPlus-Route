import pytest
import torch
from logic.src.pipeline.rl.common.post_processing import EfficiencyOptimizer, calculate_efficiency, decode_routes


class TestPostProcessing:
    def test_decode_routes_simple(self):
        # [0, 1, 2, 0, 0] -> [1, 2]
        actions = torch.tensor([[0, 1, 2, 0, 0], [0, 3, 0, 0, 0]])
        routes = decode_routes(actions, num_nodes=3)
        assert len(routes) == 2
        assert routes[0] == [1, 2]
        assert routes[1] == [3]

    def test_decode_routes_empty(self):
        actions = torch.tensor([[0, 0, 0]])
        routes = decode_routes(actions, 3)
        assert routes[0] == []

    def test_calculate_efficiency_basic(self):
        # 3 Nodes (0=Depot, 1, 2).
        # Route: 1 -> 2.
        # Demand: [0, 10, 20]. Capacity: 100.
        # Dist: 0-1 (10), 1-2 (5), 2-0 (10). Total 25.

        routes = [[1, 2]]
        demand = torch.tensor([0.0, 10.0, 20.0])
        dist_matrix = torch.zeros((3, 3))
        # Populate dists
        dist_matrix[0, 1] = 10.0
        dist_matrix[1, 0] = 10.0
        dist_matrix[1, 2] = 5.0
        dist_matrix[2, 1] = 5.0
        dist_matrix[0, 2] = 10.0
        dist_matrix[2, 0] = 10.0  # 2-0 is 10

        eff = calculate_efficiency(routes, dist_matrix, demand, capacity=100.0)

        10.0 + 5.0 + 10.0  # 25

        assert eff == pytest.approx(30.0 / 25.0)

    def test_calculate_efficiency_empty_routes(self):
        routes = [[], [1]]
        demand = torch.tensor([0.0, 10.0])
        dist_matrix = torch.zeros((2, 2))
        dist_matrix[0, 1] = 5.0
        dist_matrix[1, 0] = 5.0

        eff = calculate_efficiency(routes, dist_matrix, demand, 100)

        # Route 1: [] -> 0 waste, 0 dist?
        # Route 2: [1] -> 10 waste. Dist 0->1->0 = 10.
        # Total waste 10. Total dist 10. Eff 1.0.
        assert eff == pytest.approx(1.0)

    def test_calculate_efficiency_zero_dist(self):
        # Should guard div by zero
        routes = [[]]
        demand = torch.tensor([0.0])
        dist_matrix = torch.zeros((1, 1))

        eff = calculate_efficiency(routes, dist_matrix, demand, 100)
        assert eff == 0.0

    def test_optimizer_stub(self):
        opt = EfficiencyOptimizer("problem")
        routes = [torch.tensor([1, 2])]
        optimized = opt.optimize(routes)
        assert optimized == routes
