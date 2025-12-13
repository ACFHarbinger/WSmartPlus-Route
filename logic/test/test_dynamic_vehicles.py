
import pytest
import numpy as np
from logic.src.or_policies.multi_vehicle import find_routes

def test_find_routes_excess_vehicles():
    # Scenario: Needs 3 vehicles (capacity 2, demand 5).
    # Provide 10 vehicles.
    # Solver should use 3.
    
    # Fully connected, 10 distance everywhere except 0 at diagonal
    dist_matrix = np.full((6, 6), 10, dtype=np.int32)
    np.fill_diagonal(dist_matrix, 0)
    
    # Demands: 1 for 1..5. Depot demand irrelevant (shouldn't be in input array based on new structure?)
    # Wait, new structure in `multi_vehicle.py`: `demands[original_idx - 1]`.
    # Original indices: 1, 2, 3, 4, 5.
    # demands array should len 5.
    demands = np.array([1, 1, 1, 1, 1])
    max_capacity = 2
    to_collect = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    n_vehicles = 10 # Excess
    
    tour, cost = find_routes(dist_matrix, demands, max_capacity, to_collect, n_vehicles)
    
    # Calculate trips by counting 0s (excluding start/end if stripped? No, find_routes returns [0, ..., 0])
    # [0, 1, 2, 0, 3, 4, 0, 5, 0] -> 3 trips?
    # Count of 0s: 
    # if tour is [0, 1, 0] -> 2 zeros, 1 trip.
    # if tour is [0, 1, 0, 2, 0] -> 3 zeros, 2 trips.
    # Trips = zeros - 1
    
    zeros = tour.count(0)
    trips = zeros - 1
    
    # Capacity 2, Demand 5. Must satisfy ceil(5/2) = 3 trips.
    assert trips >= 3
    # Should not use 10 vehicles (9 zeros?) unless beneficial (cost is uniform so likely min vehicles)
    assert trips < 10
    
