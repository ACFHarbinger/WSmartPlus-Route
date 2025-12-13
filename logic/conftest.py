import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def hgs_inputs():
    # Simple scenario: Depot + 4 nodes
    # Distances: linear 0--1--2--3--4
    dist_matrix = [
        [0, 10, 20, 30, 40],
        [10, 0, 10, 20, 30],
        [20, 10, 0, 10, 20],
        [30, 20, 10, 0, 10],
        [40, 30, 20, 10, 0]
    ]

    demands = {1: 10, 2: 10, 3: 10, 4: 10}
    capacity = 100
    R = 1.0
    C = 1.0

    global_must_go = {2, 4}  # Must go bins
    local_to_global = {0: 1, 1: 2, 2: 3, 3: 4}  # Linear mapping

    vrpp_tour_global = [2, 4, 1, 3]  # Some initial tour

    return dist_matrix, demands, capacity, R, C, global_must_go, local_to_global, vrpp_tour_global
