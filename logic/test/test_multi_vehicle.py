
import pytest
import numpy as np
from logic.src.or_policies.multi_vehicle import find_routes, find_routes_ortools

def test_find_routes_basic():
    # 0 is depot. 1, 2, 3 are bins.
    # to_collect = [1, 2, 3] (indices in 0-based system, 0 is depot)
    # Distances: symmetric, simple.
    
    # 0 <-> 1: 10
    # 1 <-> 2: 10
    # 2 <-> 3: 10
    # 3 <-> 0: 10
    # Should result in 0-1-2-3-0 if capacity allows.
    
    dist_matrix = np.array([
        [0, 10, 20, 10], # 0
        [10, 0, 10, 20], # 1
        [20, 10, 0, 10], # 2
        [10, 20, 10, 0]  # 3
    ], dtype=np.int32)
    
    # Demands: 1 for others. No depot demand in input array.
    demands = np.array([1, 1, 1])
    max_capacity = 5
    to_collect = np.array([1, 2, 3], dtype=np.int32)
    n_vehicles = 1
    depot = 0
    
    tour, cost = find_routes(dist_matrix, demands, max_capacity, to_collect, n_vehicles, depot=depot)
    
    # Check structure
    assert isinstance(tour, list)
    assert tour[0] == 0 or tour[-1] == 0
    # Check if all nodes visited
    assert set(tour) == {0, 1, 2, 3}
    
    # Cost should be 10+10+10+10 = 40
    # Or maybe 30 if 0-1-2-3-0? 0-1(10), 1-2(10), 2-3(10), 3-0(10) -> 40.
    # Depending on how cost is calculated.
    
def test_find_routes_two_vehicles():
    # Make it impossible for 1 vehicle (capacity 2, total demand 4)
    dist_matrix = np.zeros((5, 5), dtype=np.int32)
    # Just need it to run without error and respect constraints roughly
    # 0, 1, 2, 3, 4
    for i in range(5):
        for j in range(5):
            if i != j:
                dist_matrix[i][j] = 10
                
    demands = np.array([1, 1, 1, 1])
    max_capacity = 2
    to_collect = np.array([1, 2, 3, 4], dtype=np.int32)
    n_vehicles = 2
    depot = 0
    
    tour, cost = find_routes(dist_matrix, demands, max_capacity, to_collect, n_vehicles, depot=depot)
    
    # Should have at least one return to depot in middle
    # tour like [0, 1, 2, 0, 3, 4, 0]
    zero_counts = tour.count(0)
    assert zero_counts >= 3 
    
def test_find_routes_unlimited_vehicles():
    # Use 0 vehicles -> unlimited.
    # Scenario: Needs 3 vehicles (capacity 2, demand 5).
    dist_matrix = np.full((6, 6), 10, dtype=np.int32)
    np.fill_diagonal(dist_matrix, 0)
    
    demands = np.array([1, 1, 1, 1, 1])
    max_capacity = 2
    to_collect = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    n_vehicles = 0 # Unlimited
    depot = 0
    
    tour, cost = find_routes(dist_matrix, demands, max_capacity, to_collect, n_vehicles, depot=depot)
    
    # Should calculate trips correctly
    zeros = tour.count(0)
    trips = zeros - 1
    assert trips >= 3
    
def test_find_routes_ortools_basic():
    # Same basic test but for OR-Tools
    dist_matrix = np.array([
        [0, 10, 10, 10], # 0 (Depot)
        [10, 0, 10, 20], # 1
        [10, 10, 0, 10], # 2
        [10, 20, 10, 0]  # 3
    ], dtype=np.int32)
    
    demands = np.array([1, 1, 1])
    max_capacity = 5
    to_collect = np.array([1, 2, 3], dtype=np.int32)
    n_vehicles = 1
    
    tour, cost = find_routes_ortools(dist_matrix, demands, max_capacity, to_collect, n_vehicles)
    
    # Should visit all.
    assert 1 in tour
    assert 2 in tour
    assert 3 in tour
    assert tour[0] == 0
    assert tour[-1] == 0
    
    # Test unlimited vehicles for OR-Tools
    n_vehicles_unlimited = 0
    max_capacity_small = 1 # Force split
    # demands 1, capacity 1 -> 3 trips
    tour_u, cost_u = find_routes_ortools(dist_matrix, demands, max_capacity_small, to_collect, n_vehicles_unlimited)
    
    zeros = tour_u.count(0)
    # 3 trips -> S-1-E-S-2-E-S-3-E -> [0, 1, 0, 2, 0, 3, 0] -> 4 zeros -> 3 trips.
    # OR-Tools flat tour construction logic should be verified.
    # Logic: [0, 1, 0] + [2, 0] -> [0, 1, 0, 2, 0] if merging...
    # My logic: 
    # for t in tours: if not flat: add 0. add nodes. add 0.
    # [0, 1, 0] -> flat: [0, 1, 0]
    # [0, 2, 0] -> flat: [0, 1, 0, 2, 0]

    
