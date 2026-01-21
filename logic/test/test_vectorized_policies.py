"""
Unit tests for vectorized policy implementations (Local Search, HGS, Split).
"""
import pytest
import torch

from logic.src.pipeline.reinforcement_learning.policies.hgs_vectorized import (
    VectorizedHGS,
    VectorizedPopulation,
    calc_broken_pairs_distance,
    vectorized_ordered_crossover,
)
from logic.src.pipeline.reinforcement_learning.policies.local_search import (
    vectorized_relocate,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_three_opt,
    vectorized_two_opt,
    vectorized_two_opt_star,
)
from logic.src.pipeline.reinforcement_learning.policies.split_algorithm import (
    vectorized_linear_split,
)


class TestVectorizedLocalSearch:
    """Tests for vectorized local search heuristics."""

    @pytest.mark.unit
    def test_vectorized_two_opt(self, v_device):
        """Test basic 2-opt improvement."""
        pts = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=v_device)
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_matrix = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        tours = torch.tensor([[0, 1, 2, 3, 0]], device=v_device)
        optimized_tours = vectorized_two_opt(tours.clone(), dist_matrix, max_iterations=50)

        def calc_cost(t, d):
            """Manual cost calculation for verification."""
            c = 0
            for k in range(len(t) - 1):
                c += d[0, t[k], t[k + 1]]
            return c

        orig_cost = calc_cost(tours[0], dist_matrix)
        opt_cost = calc_cost(optimized_tours[0], dist_matrix)
        assert opt_cost < orig_cost - 0.1
        assert abs(opt_cost - 4.0) < 1e-5

    @pytest.mark.unit
    def test_vectorized_three_opt(self, v_device):
        """Test 3-opt improvement."""
        pts = torch.tensor(
            [[0, 0], [2, 0], [2, 2], [0, 2], [1, 3], [-1, 3]],
            dtype=torch.float32,
            device=v_device,
        )
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist_matrix = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        tours = torch.tensor([[0, 1, 3, 5, 2, 4, 0]], device=v_device)

        def calc_cost(t, d):
            """Manual cost calculation for verification."""
            c = 0
            for k in range(len(t) - 1):
                c += d[0, t[k], t[k + 1]]
            return c

        orig_cost = calc_cost(tours[0], dist_matrix)
        optimized_tours = vectorized_three_opt(tours.clone(), dist_matrix, max_iterations=200)
        opt_cost = calc_cost(optimized_tours[0], dist_matrix)

        assert opt_cost <= orig_cost + 1e-5
        t = optimized_tours[0]
        assert t[0] == 0 and t[-1] == 0
        assert len(torch.unique(t[1:-1])) == 5

    @pytest.mark.unit
    def test_vectorized_swap(self, v_device, v_mock_dist_matrix):
        """Test vectorized swap operator."""
        tours = torch.tensor([[0, 1, 2, 3, 0]], device=v_device)
        dist = v_mock_dist_matrix.unsqueeze(0)
        new_tours = vectorized_swap(tours.clone(), dist, max_iterations=200)
        assert new_tours.shape == tours.shape

    @pytest.mark.unit
    def test_vectorized_relocate(self, v_device, v_mock_dist_matrix):
        """Test vectorized relocate operator."""
        tours = torch.tensor([[0, 1, 2, 3, 0]], device=v_device)
        dist = v_mock_dist_matrix.unsqueeze(0)
        new_tours = vectorized_relocate(tours.clone(), dist, max_iterations=200)
        assert new_tours.shape == tours.shape

    @pytest.mark.unit
    def test_vectorized_two_opt_star(self, v_device):
        """Test vectorized two-opt* operator."""
        pts = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 1.0], [1.0, 2.0]],
            device=v_device,
        )
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        tours = torch.tensor([[0, 1, 4, 0, 3, 2, 0]], device=v_device)
        new_tours = vectorized_two_opt_star(tours.clone(), dist, max_iterations=100)
        assert new_tours.shape == tours.shape

    @pytest.mark.unit
    def test_vectorized_swap_star(self, v_device):
        """Test vectorized swap* operator."""
        pts = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 1.0], [1.0, 2.0]],
            device=v_device,
        )
        diff = pts.unsqueeze(1) - pts.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(dim=2)).unsqueeze(0)
        tours = torch.tensor([[0, 1, 4, 0, 3, 2, 0]], device=v_device)
        new_tours = vectorized_swap_star(tours.clone(), dist, max_iterations=100)
        assert new_tours.shape == tours.shape


class TestVectorizedPolicies:
    """Tests for the Hybrid Genetic Search vectorized implementation."""

    @pytest.mark.unit
    def test_vectorized_linear_split(self, v_device, v_sample_data):
        """Test vectorized linear split."""
        dist, demands = v_sample_data
        tours = torch.tensor([[1, 2, 3, 4, 5]], device=v_device)
        dist_b = dist.unsqueeze(0)
        demands_b = demands.unsqueeze(0)
        capacity = 2.0
        routes, costs = vectorized_linear_split(tours, dist_b, demands_b, capacity)
        assert len(routes) == 1
        assert costs.shape == (1,)

    @pytest.mark.unit
    def test_broken_pairs_distance(self, v_device):
        """Test broken pairs distance diversity."""
        B, P = 1, 2
        pop = torch.tensor([[[1, 2, 3, 4], [1, 2, 4, 3]]], device=v_device)
        diversity = calc_broken_pairs_distance(pop)
        assert diversity.shape == (B, P)
        assert diversity[0, 0] > 0

    @pytest.mark.unit
    def test_vectorized_ordered_crossover(self, v_device):
        """Test OX1 crossover."""
        B, N = 2, 5
        p1 = torch.stack([torch.randperm(N) + 1 for _ in range(B)]).to(v_device)
        p2 = torch.stack([torch.randperm(N) + 1 for _ in range(B)]).to(v_device)
        offspring = vectorized_ordered_crossover(p1, p2)
        assert offspring.size() == (B, N)
        assert len(torch.unique(offspring[0])) == N

    @pytest.mark.unit
    def test_vectorized_hgs_solve(self, v_device, v_sample_data):
        """Test full HGS solve run."""
        dist, demands = v_sample_data
        B = 1
        dist_b = dist.unsqueeze(0)
        demands_b = demands.unsqueeze(0)
        hgs = VectorizedHGS(dist_b, demands_b, vehicle_capacity=3.0, device=v_device)
        init_sol = torch.tensor([[1, 2, 3, 4, 5]], device=v_device)
        routes, costs = hgs.solve(init_sol, n_generations=5, population_size=10)
        assert len(routes) == B
        assert costs.shape == (B,)
        assert costs[0] < 1e9


class TestVectorizedPopulation:
    """Detailed tests for VectorizedPopulation management."""

    @pytest.mark.unit
    def test_population_add_individuals_selection(self, v_device):
        """Test that add_individuals keeps only best survivors."""

        max_size = 3
        pop = VectorizedPopulation(size=max_size, device=v_device, alpha_diversity=0.0)  # alpha=0 -> only cost matters

        # Initial pop (size 2)
        initial_pop = torch.tensor([[[1, 2, 3, 4], [4, 3, 2, 1]]], device=v_device)
        initial_costs = torch.tensor([[10.0, 20.0]], device=v_device)  # 10 is best
        pop.initialize(initial_pop, initial_costs)
        assert pop.population.size(1) == 2

        # Add 3 more (total 5 > max_size 3)
        candidates = torch.tensor([[[2, 3, 4, 1], [3, 4, 1, 2], [1, 4, 2, 3]]], device=v_device)
        candidate_costs = torch.tensor([[5.0, 15.0, 30.0]], device=v_device)  # 5 is now best

        # Expected survivors (sorted by cost since alpha=0): 5.0, 10.0, 15.0
        pop.add_individuals(candidates, candidate_costs)

        assert pop.population.size(1) == max_size
        assert torch.allclose(pop.costs.sort().values, torch.tensor([[5.0, 10.0, 15.0]], device=v_device))

    @pytest.mark.unit
    def test_population_diversity_impact(self, v_device):
        """Test that diversity (alpha_diversity) affects biased fitness."""
        # 3 individuals:
        # A: Cost 10, Diversity Low
        # B: Cost 11, Diversity High
        # If alpha is 100, B should be better than A despite slightly higher cost.

        pop = VectorizedPopulation(size=10, device=v_device, alpha_diversity=10.0)

        # A: [1, 2, 3, 4], B: [1, 3, 2, 4] (High distance between them - different edges)
        pop_data = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 3, 2, 4]]], device=v_device)
        costs = torch.tensor([[10.0, 10.1, 12.0]], device=v_device)

        # With alpha=0, biased_fitness matches cost ranks: Ind 0 (0), Ind 1 (1), Ind 2 (2)
        pop.alpha_diversity = 0.0
        pop.initialize(pop_data, costs)
        assert pop.biased_fitness[0, 0] < pop.biased_fitness[0, 1] < pop.biased_fitness[0, 2]

        # With high alpha, Ind 2 (most diverse) should get much better rank than Ind 1
        pop.alpha_diversity = 100.0
        pop.compute_biased_fitness()
        # Ind 2 is more diverse than Ind 1. Ind 1 and 0 are same, so they have low diversity.
        # Diversity Rank of Ind 2 should be 0 (best). Ind 0 and 1 will have rank 1 or 2.
        assert pop.biased_fitness[0, 2] < pop.biased_fitness[0, 1]

    @pytest.mark.unit
    def test_population_get_parents_tournament(self, v_device):
        """Test parent selection (tournament shape and randomness)."""
        B, P, N = 2, 10, 5
        pop = VectorizedPopulation(size=P, device=v_device)
        population = torch.stack([torch.randperm(N) + 1 for _ in range(B * P)]).view(B, P, N).to(v_device)
        costs = torch.rand((B, P), device=v_device)
        pop.initialize(population, costs)

        n_off = 4
        p1, p2 = pop.get_parents(n_offspring=n_off)

        assert p1.shape == (B, n_off, N)
        assert p2.shape == (B, n_off, N)

        # Check that nodes are valid
        assert (p1 > 0).all() and (p1 <= N).all()
