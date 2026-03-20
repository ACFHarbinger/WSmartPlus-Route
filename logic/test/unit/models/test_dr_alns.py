"""
Tests for DR-ALNS implementation.
"""

import numpy as np
import pytest
import torch

from logic.src.envs.dr_alns import DRALNSEnv
from logic.src.models.core.dr_alns import DRALNSPPOAgent, DRALNSSolver


@pytest.fixture
def mock_vrpp_instance():
    """Create a small mock VRPP instance."""
    n_nodes = 5
    rng = np.random.RandomState(42)
    locations = rng.rand(n_nodes + 1, 2) * 100
    dist_matrix = np.zeros((n_nodes + 1, n_nodes + 1))
    for i in range(n_nodes + 1):
        for j in range(n_nodes + 1):
            dist_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])

    wastes = {i: 5.0 for i in range(1, n_nodes + 1)}
    capacity = 20.0

    return {
        "dist_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": capacity,
        "R": 1.0,
        "C": 0.1,
    }


def test_dralns_env(mock_vrpp_instance):
    """Test DRALNSEnv basic functionality."""
    env = DRALNSEnv(
        max_iterations=10,
        n_destroy_ops=3,
        n_repair_ops=2,
    )

    obs, info = env.reset(options={"instance": mock_vrpp_instance})
    assert obs.shape == (7,)
    assert "best_profit" in info

    action = np.array([0, 0, 0, 0])  # destroy_idx, repair_idx, severity_idx, temp_idx
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (7,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert "current_profit" in info


def test_ppo_agent():
    """Test DRALNSPPOAgent forward pass and action sampling."""
    agent = DRALNSPPOAgent(state_dim=7, hidden_dim=16)
    state = torch.randn(1, 7)

    actions, log_probs, value = agent.get_action(state)

    assert "destroy" in actions
    assert "repair" in actions
    assert "severity" in actions
    assert "temp" in actions
    assert value.shape == (1, 1)


def test_dralns_solver(mock_vrpp_instance):
    """Test DRALNSSolver execution."""
    agent = DRALNSPPOAgent(state_dim=7, hidden_dim=16)
    solver = DRALNSSolver(
        dist_matrix=mock_vrpp_instance["dist_matrix"],
        wastes=mock_vrpp_instance["wastes"],
        capacity=mock_vrpp_instance["capacity"],
        R=mock_vrpp_instance["R"],
        C=mock_vrpp_instance["C"],
        agent=agent,
        max_iterations=5,
    )

    best_routes, best_profit, best_cost = solver.solve()

    assert isinstance(best_routes, list)
    assert isinstance(best_profit, float)
    assert isinstance(best_cost, float)
