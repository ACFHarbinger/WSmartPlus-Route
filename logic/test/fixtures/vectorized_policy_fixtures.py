"""
Fixtures for vectorized policy unit tests.
"""
import pytest
import torch


@pytest.fixture
def v_device():
    """Returns the device for vectorized tests."""
    return "cpu"


@pytest.fixture
def v_mock_dist_matrix(v_device):
    """Fixture for a mock distance matrix for vectorized tests."""
    # Simple Euclidean distance for points (0,0), (0,1), (1,0), (1,1)
    pts = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=v_device)
    diff = pts.unsqueeze(1) - pts.unsqueeze(0)
    dist = torch.sqrt((diff**2).sum(dim=2))
    return dist


@pytest.fixture
def v_sample_data(v_device):
    """Fixture for sample data (distance matrix and demands) for vectorized tests."""
    # 5 nodes + depot (0)
    pts = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        device=v_device,
    )
    diff = pts.unsqueeze(1) - pts.unsqueeze(0)
    dist = torch.sqrt((diff**2).sum(dim=2))
    demands = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=v_device)
    return dist, demands
