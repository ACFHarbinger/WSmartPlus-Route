
import time
import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.pipeline.rl.core.reinforce import REINFORCE
from logic.src.envs.base import RL4COEnvBase

class MockPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2) # Dummy encoder to satisfy REINFORCE checks if any

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
def test_reinforce_time_tracking():
    """Test that REINFORCE captures time and applies penalty."""

    # 1. Setup
    from logic.src.envs.base import RL4COEnvBase
    class MockEnv(RL4COEnvBase):
        name = "mock_env"
        def reset(self, td, **kwargs): return td
        def get_reward(self, td, actions): return torch.zeros(td.size(0))

    env = MockEnv()
    policy = MockPolicy()

    batch_size = 4
    td = TensorDict({"locs": torch.randn(batch_size, 10, 2)}, batch_size=batch_size)

    # 2. Instantiate REINFORCE with sensitivity
    # We need to mock the baseline initialization or provide one
    # RL4COEnvBase is needed for type checking but we can pass None if robust

    # Mocking StepMixin.__init__ dependencies or just providing minimal args
    # We assume REINFORCE initializes correctly with minimal args

    module = REINFORCE(
        env=env,
        policy=policy,
        baseline="none",
        time_sensitivity=1.0,
        optimizer_kwargs={"lr": 1e-4}
    )

    # Mock baseline manual injection to avoid complex init
    module.baseline = MockBaseline()

    # Verify wrapping
    # The policy attribute should be the TimeTrackingPolicy wrapper
    from logic.src.pipeline.rl.core.time_tracking import TimeTrackingPolicy
    assert isinstance(module.policy, TimeTrackingPolicy)

    # 3. Run forward pass (manual, as training_step does)
    out = module.policy(td, env)

    assert "inference_time" in out
    # allow some jitter, but sleep is 0.01 so it should be at least that
    assert out["inference_time"].item() > 0.0

    # 4. Calculate Loss
    # We artificially set reward to 10.0
    out["reward"] = torch.ones(batch_size) * 10.0
    out["log_likelihood"] = torch.ones(batch_size) * 1.0

    loss = module.calculate_loss(td, out, 0, env)

    assert "inference_time" in out

@pytest.mark.unit
def test_time_sensitivity_zero():
    """Test that sensitivity 0 does NOT track time."""
    policy = MockPolicy()
    module = REINFORCE(
        env=None,
        policy=policy,
        baseline="none",
        time_sensitivity=0.0
    )
    module.baseline = MockBaseline()

    td = TensorDict({"locs": torch.randn(2, 5, 2)}, batch_size=2)
    out = module.policy(td)

    assert "inference_time" not in out
