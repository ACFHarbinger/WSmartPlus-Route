
import time
import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.pipeline.rl.core.reinforce import REINFORCE
from logic.src.pipeline.rl.core.time_tracking import TimeOptimizedREINFORCE
from logic.src.envs.base import RL4COEnvBase

class MockPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)

    def forward(self, td, env=None, **kwargs):
        # Simulate work
        time.sleep(0.01)

        batch_size = td.size(0)
        # Return dummy outputs
        return {
            "reward": torch.zeros(batch_size, device=td.device),
            "log_likelihood": torch.zeros(batch_size, device=td.device),
            "entropy": torch.zeros(batch_size, device=td.device),
        }

class MockBaseline(nn.Module):
    def __init__(self):
        super().__init__()

    def eval(self, td, reward, env=None):
        return torch.zeros_like(reward)

@pytest.mark.unit
def test_reinforce_base_no_time():
    """Test that base REINFORCE does NOT track time."""
    # Mock dependencies
    class MockEnv(RL4COEnvBase):
        name = "mock_env"
        def reset(self, td, **kwargs): return td
        def get_reward(self, td, actions): return torch.zeros(td.size(0))

    env = MockEnv()
    policy = MockPolicy()

    module = REINFORCE(
        env=env,
        policy=policy,
        baseline="none"
    )
    module.baseline = MockBaseline()

    td = TensorDict({"locs": torch.randn(2, 5, 2)}, batch_size=2)
    # The policy is not wrapped
    out = module.policy(td)

    assert "inference_time" not in out

@pytest.mark.unit
def test_time_optimized_reinforce():
    """Test that TimeOptimizedREINFORCE captures time and applies penalty."""

    class MockEnv(RL4COEnvBase):
        name = "mock_env"
        def reset(self, td, **kwargs): return td
        def get_reward(self, td, actions): return torch.zeros(td.size(0))

    env = MockEnv()
    policy = MockPolicy()
    batch_size = 4
    td = TensorDict({"locs": torch.randn(batch_size, 10, 2)}, batch_size=batch_size)

    module = TimeOptimizedREINFORCE(
        env=env,
        policy=policy,
        baseline="none",
        time_sensitivity=1.0,
        optimizer_kwargs={"lr": 1e-4}
    )

    module.baseline = MockBaseline()

    # Verify wrapping
    from logic.src.models.attention_model.time_tracking_policy import TimeTrackingPolicy
    assert isinstance(module.policy, TimeTrackingPolicy)

    # Run forward pass
    out = module.policy(td, env)

    assert "inference_time" in out
    assert out["inference_time"].item() > 0.0

    # Calculate Loss
    out["reward"] = torch.ones(batch_size) * 10.0
    out["log_likelihood"] = torch.ones(batch_size) * 1.0

    # This should call TimeOptimizedREINFORCE.calculate_loss
    loss = module.calculate_loss(td, out, 0, env)

    assert "inference_time" in out

@pytest.mark.unit
def test_time_optimized_reinforce_zero():
    """Test that TimeOptimizedREINFORCE with sensitivity 0 does NOT track time."""
    policy = MockPolicy()
    module = TimeOptimizedREINFORCE(
        env=None,
        policy=policy,
        baseline="none",
        time_sensitivity=0.0
    )
    module.baseline = MockBaseline()

    td = TensorDict({"locs": torch.randn(2, 5, 2)}, batch_size=2)
    out = module.policy(td)

    assert "inference_time" not in out
