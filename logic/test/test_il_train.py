
import pytest
import torch
import time
from logic.src.pipeline.reinforcement_learning.policies.hgs_vectorized import (
    VectorizedHGS, 
    VectorizedPopulation, 
    calc_broken_pairs_distance,
    vectorized_swap,
    vectorized_relocate,
    vectorized_two_opt_star,
    vectorized_swap_star,
    vectorized_ordered_crossover,
    vectorized_linear_split
)
from logic.src.pipeline.reinforcement_learning.core.post_processing import local_search_2opt_vectorized


class TestHGS:
    @pytest.fixture
    def device(self):
        return 'cpu'

    @pytest.fixture
    def mock_dist_matrix(self, device):
        # Simple Euclidean distance for points (0,0), (0,1), (1,0), (1,1)
        pts = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2))
        return dist

    @pytest.fixture
    def sample_data(self, device):
        # 5 nodes + depot (0)
        pts = torch.tensor([[0.,0.], [1.,0.], [2.,0.], [3.,0.], [4.,0.], [5.,0.]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2))
        demands = torch.tensor([0., 1., 1., 1., 1., 1.], device=device)
        return dist, demands

    def test_broken_pairs_distance(self, device):
        B, P, N = 2, 4, 3
        pop = torch.tensor([[1, 2, 3]], device=device).expand(B, P, N)
        diversity = calc_broken_pairs_distance(pop)
        assert torch.allclose(diversity, torch.zeros_like(diversity))
        
        pop3 = torch.tensor([
            [1, 2, 3, 4], 
            [1, 2, 4, 3]
        ], device=device).unsqueeze(0)
        
        diversity = calc_broken_pairs_distance(pop3)
        assert torch.isclose(diversity[0, 0], torch.tensor(0.5, device=device))
        assert torch.isclose(diversity[0, 1], torch.tensor(0.5, device=device))

    def test_vectorized_swap(self, device, mock_dist_matrix):
        tours = torch.tensor([[0, 1, 2, 3, 0]], device=device)
        dist = mock_dist_matrix.unsqueeze(0)
        new_tours = vectorized_swap(tours.clone(), dist, max_iterations=200)
        
        def calc_cost(t):
            c = 0
            for k in range(len(t)-1):
                c += dist[0, t[k], t[k+1]]
            return c
            
        orig_cost = calc_cost(tours[0])
        new_cost = calc_cost(new_tours[0])
        assert new_cost <= orig_cost + 1e-5

    def test_vectorized_relocate(self, device, mock_dist_matrix):
        tours = torch.tensor([[0, 1, 2, 3, 0]], device=device)
        dist = mock_dist_matrix.unsqueeze(0)
        new_tours = vectorized_relocate(tours.clone(), dist, max_iterations=200)
        orig_cost = 4.414 # Known cost for 0-1-2-3-0
        
        def calc_cost(t):
            c = 0
            for k in range(len(t)-1):
                c += dist[0, t[k], t[k+1]]
            return c
            
        new_cost = calc_cost(new_tours[0])
        assert new_cost <= orig_cost + 1e-5

    def test_vectorized_two_opt_star(self, device):
        # Crossing path scenario
        pts = torch.tensor([[0.,0.], [0.,1.], [0.,2.], [1.,1.], [1.,2.]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        
        tours = torch.tensor([[0, 1, 4, 0, 3, 2, 0]], device=device)
        new_tours = vectorized_two_opt_star(tours.clone(), dist, max_iterations=100)
        
        def calc_cost(t):
            c = 0
            for k in range(len(t)-1):
                c += dist[0, t[k], t[k+1]]
            return c
            
        orig = calc_cost(tours[0])
        new_c = calc_cost(new_tours[0])
        
        assert new_c < orig - 0.1
        t = new_tours[0]
        nodes = t[t!=0]
        assert len(nodes) == 4
        assert set(nodes.tolist()) == {1, 2, 3, 4}

    def test_vectorized_swap_star(self, device):
        pts = torch.tensor([[0.,0.], [0.,1.], [0.,2.], [1.,1.], [1.,2.]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        
        tours = torch.tensor([[0, 1, 4, 0, 3, 2, 0]], device=device)
        new_tours = vectorized_swap_star(tours.clone(), dist, max_iterations=100)
        
        def calc_cost(t):
            c = 0
            for k in range(len(t)-1):
                c += dist[0, t[k], t[k+1]]
            return c
            
        orig = calc_cost(tours[0])
        new_c = calc_cost(new_tours[0])
        
        assert new_c < orig - 0.1
        t = new_tours[0]
        nodes = t[t!=0]
        assert len(nodes) == 4
        assert set(nodes.tolist()) == {1, 2, 3, 4}

    def test_population_diversity(self, device):
        B, P, N = 1, 10, 5
        pop = VectorizedPopulation(size=5, device=device, alpha_diversity=1.0)
        population = torch.stack([torch.randperm(N, device=device) + 1 for _ in range(P)]).unsqueeze(0)
        costs = torch.rand((B, P), device=device)
        pop.initialize(population, costs)
        
        assert pop.biased_fitness is not None
        assert pop.diversity_scores is not None
        assert pop.diversity_scores.shape == (B, P)
        assert pop.biased_fitness.shape == (B, P)

    def test_vectorized_ordered_crossover(self):
        B, N = 4, 10
        p1 = torch.stack([torch.randperm(N) + 1 for _ in range(B)])
        p2 = torch.stack([torch.randperm(N) + 1 for _ in range(B)])
        offspring = vectorized_ordered_crossover(p1, p2)
        assert offspring.size() == (B, N)
        for b in range(B):
            nodes = offspring[b]
            assert len(torch.unique(nodes)) == N
            assert nodes.min() == 1
            assert nodes.max() == N

    def test_vectorized_linear_split(self, device, sample_data):
        dist, demands = sample_data
        # 5 nodes + depot. Demands = 1 each.
        # Capacity = 2.
        # Expected split:
        # Route 1: 1-2 (Load 2)
        # Route 2: 3-4 (Load 2)
        # Route 3: 5 (Load 1)
        # Or similar.
        
        # Giant tour: 1-2-3-4-5
        tours = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        B = 1
        dist_b = dist.unsqueeze(0)
        demands_b = demands.unsqueeze(0)
        
        capacity = 2.0
        
        routes, costs = vectorized_linear_split(tours, dist_b, demands_b, capacity)
        
        # Check output structure
        assert len(routes) == B
        # routes[0] is a flat list [0, n, n, 0, n, n, 0...]
        r_flat = routes[0]
        assert isinstance(r_flat, list)
        
        # Parse into individual routes
        parsed_routes = []
        current_route = [0]
        for node in r_flat[1:]:
            current_route.append(node)
            if node == 0:
                if len(current_route) > 2: # Ignore 0-0 sequence if any
                    parsed_routes.append(current_route)
                current_route = [0]
                
        # With capacity 2 and 5 items, we expect at least 3 routes.
        assert len(parsed_routes) == 3
        
        # Check validity
        for r in parsed_routes:
            # r is [0, n1, n2, ..., 0]
            assert r[0] == 0 and r[-1] == 0
            nodes = r[1:-1]
            load = demands[nodes].sum()
            assert load <= capacity + 1e-5
            
        # Check total nodes cover
        all_nodes = []
        for r in parsed_routes:
            all_nodes.extend(r[1:-1])
        all_nodes = sorted(all_nodes)
        assert all_nodes == [1, 2, 3, 4, 5]


    def test_time_limit(self, device, sample_data):
        dist, demands = sample_data
        B = 2
        dist_b = dist.unsqueeze(0).expand(B, -1, -1)
        demands_b = demands.unsqueeze(0).expand(B, -1)
        
        hgs = VectorizedHGS(dist_b, demands_b, vehicle_capacity=10.0, time_limit=10.0, device=device)
        init_sol = torch.stack([torch.randperm(5, device=device) + 1 for _ in range(B)])
        
        start_t = time.time()
        hgs.solve(init_sol, n_generations=10000, population_size=10, time_limit=0.2)
        end_t = time.time()
        assert (end_t - start_t) < 1.0

    def test_max_vehicles_constraint(self, device, sample_data):
        dist, demands = sample_data
        B = 1
        dist_b = dist.unsqueeze(0)
        demands_b = demands.unsqueeze(0)
        tours = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        
        hgs = VectorizedHGS(dist_b, demands_b, vehicle_capacity=3.0, device=device)
        routes, costs = hgs.solve(tours, n_generations=1, population_size=5, max_vehicles=2)
        assert costs[0] < float('inf')
        
        routes_inf, costs_inf = hgs.solve(tours, n_generations=1, population_size=5, max_vehicles=1)
        assert costs_inf[0] == float('inf') or costs_inf[0] > 1e9

    def test_stagnation_restart_runs(self, device, sample_data):
        dist, demands = sample_data
        B = 1
        dist_b = dist.unsqueeze(0)
        demands_b = demands.unsqueeze(0)
        hgs = VectorizedHGS(dist_b, demands_b, vehicle_capacity=10.0, device=device)
        init_sol = torch.stack([torch.randperm(5, device=device) + 1 for _ in range(B)])
        
        # Run enough gens to trigger restart (50 is threshold)
        hgs.solve(init_sol, n_generations=60, population_size=10, elite_size=2)


class TestVectorized2OptIL:
    @pytest.fixture
    def device(self):
        return 'cpu'

    def test_2opt_basic_improvement(self, device):
        # Scenario: Crossing paths
        # 0 at (0,0), 1 at (0,1), 2 at (1,0), 3 at (1,1)
        # Tour: 0 -> 1 -> 2 -> 3 -> 0
        # Edges: (0,1), (1,2), (2,3), (3,0)
        # Dist: 1 + sqrt(2) + 1 + 1 = 3 + 1.414 = 4.414
        # Optimal: 0 -> 1 -> 3 -> 2 -> 0 (Square perimeter)
        # Edges: (0,1), (1,3), (3,2), (2,0)
        # Dist: 1 + 1 + 1 + 1 = 4.0
        
        pts = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_matrix = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0) # (1, 4, 4)
        
        # Initial tour: 0-1-2-3-0. But operator expects indices?
        # If input is (B, N) or (B, N+2) depending on if depot included.
        # local_search_2opt_vectorized expects batch of tours.
        # It typically works on indices.
        
        # tour: [0, 1, 2, 3] (if depot implied?) or [0, 1, 2, 3, 0]?
        # The operator implementation looks for edges (i, i+1).
        # It requires indices to be valid for distance_matrix.
        # If N=4 (0..3), tour should just contain 0,1,2,3.
        # If it's a cycle, it usually ends with 0.
        
        tours = torch.tensor([[0, 1, 2, 3, 0]], device=device)
        
        optimized_tours = local_search_2opt_vectorized(tours.clone(), dist_matrix, max_iterations=50)
        
        def calc_cost(t, d):
            c = 0
            for k in range(len(t)-1):
                c += d[0, t[k], t[k+1]]
            return c
            
        orig_cost = calc_cost(tours[0], dist_matrix)
        opt_cost = calc_cost(optimized_tours[0], dist_matrix)
        
        assert opt_cost < orig_cost - 0.1
        assert abs(opt_cost - 4.0) < 1e-5

    def test_2opt_batch_consistency(self, device):
        # Two identical tours in batch should produce identical results
        pts = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_matrix = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        dist_batch = dist_matrix.expand(2, -1, -1)
        
        tours = torch.tensor([
            [0, 1, 2, 3, 0],
            [0, 1, 2, 3, 0]
        ], device=device)
        
        optimized = local_search_2opt_vectorized(tours.clone(), dist_batch, max_iterations=50)
        
        assert torch.all(optimized[0] == optimized[1])
        # Both should be improved
        assert (optimized[0] == torch.tensor([0, 1, 3, 2, 0], device=device)).all()

    def test_2opt_stability(self, device):
        # Optimal tour shouldn't change
        # 0 -> 1 -> 3 -> 2 -> 0 is optimal 4.0
        
        pts = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]], device=device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_matrix = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        
        optimal_tour = torch.tensor([[0, 1, 3, 2, 0]], device=device)
        
        result = local_search_2opt_vectorized(optimal_tour.clone(), dist_matrix, max_iterations=10)
        
        assert torch.all(result == optimal_tour)
