import pytest
import torch
from logic.src.models.policies.operators.random_removal import vectorized_random_removal
from logic.src.models.policies.operators.worst_removal import vectorized_worst_removal
from logic.src.models.policies.operators.greedy_insertion import vectorized_greedy_insertion
from logic.src.models.policies.operators.regret_k_insertion import vectorized_regret_k_insertion

@pytest.fixture
def data():
    B, N = 5, 20
    tours = torch.arange(1, N + 1).repeat(B, 1) # simple tours
    # Add depot 0?
    # logic seems to assume 0 is background/depot.
    # Let's make tours like: [0, 1, 2, ..., N, 0] ?
    # The operators assume "tours" as a sequence of nodes.
    # vectorized_random_removal assumes > 0 are customers.

    tours = torch.zeros((B, N + 2), dtype=torch.long)
    tours[:, 1:-1] = torch.arange(1, N + 1)

    dist_matrix = torch.rand((B, N + 1, N + 1)) # nodes 0..N

    return tours, dist_matrix

def test_vectorized_random_removal(data):
    tours, _ = data
    B, N_total = tours.shape
    n_remove = 5

    new_tours, removed = vectorized_random_removal(tours, n_remove)

    assert new_tours.shape == (B, N_total - n_remove)
    assert removed.shape == (B, n_remove)
    # Check that removed nodes are not in new_tours?
    # Since values are unique per row1..N, we can check.
    # But checking efficiently in torch...
    pass

def test_vectorized_worst_removal(data):
    tours, dist_matrix = data
    n_remove = 5
    B, N_total = tours.shape

    new_tours, removed = vectorized_worst_removal(tours, n_remove, dist_matrix)

    assert new_tours.shape == (B, N_total - n_remove)
    assert removed.shape == (B, n_remove)

def test_vectorized_greedy_insertion(data):
    tours, dist_matrix = data
    B, N_total = tours.shape
    n_rem = 3

    # Fake removed nodes
    removed_nodes = torch.tensor([[100, 101, 102]] * B, dtype=torch.long)

    # We need dist_matrix to cover these new nodes.
    # Resize dist_matrix
    ext_dist = torch.rand((B, 200, 200)) # Enough space
    ext_dist[:, :21, :21] = dist_matrix[:, :21, :21]

    new_tours = vectorized_greedy_insertion(tours, removed_nodes, ext_dist)

    assert new_tours.shape == (B, N_total + n_rem)

def test_vectorized_regret_k_insertion(data):
    tours, dist_matrix = data
    B, N_total = tours.shape
    n_rem = 3

    removed_nodes = torch.tensor([[100, 101, 102]] * B, dtype=torch.long)
    ext_dist = torch.rand((B, 200, 200))

    new_tours = vectorized_regret_k_insertion(tours, removed_nodes, ext_dist, k=2)

    assert new_tours.shape == (B, N_total + n_rem)
