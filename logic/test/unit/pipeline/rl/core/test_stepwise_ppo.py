import torch
import torch.nn as nn
from tensordict import TensorDict
import pytest
from unittest.mock import MagicMock

from logic.src.pipeline.rl.core.stepwise_ppo import StepwisePPO
from logic.src.envs.base import RL4COEnvBase

class MockPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Mock encoder that handles TensorDict input
        self.encoder = MagicMock(return_value=torch.randn(2, 10))
        self.decoder = MagicMock()

    def forward(self, td, env=None, **kwargs):
        raise NotImplementedError("Not used in checking")

class MockCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(10, 1)
    def forward(self, td):
        return torch.randn(td.batch_size[0], 1)

def test_stepwise_ppo_rollout():
    env = MagicMock(spec=RL4COEnvBase)
    env.reset.side_effect = lambda td: td.update({"done": torch.zeros(td.batch_size[0], dtype=torch.bool)})

    # Mock step to move towards done
    def mock_step(td):
        next_td = td.clone()
        next_td.update({
            "done": torch.ones(td.batch_size[0], dtype=torch.bool),
            "reward": torch.ones(td.batch_size[0], 1)
        })
        return TensorDict({"next": next_td}, batch_size=td.batch_size)

    env.step.side_effect = mock_step

    policy = MockPolicy()
    policy.decoder.return_value = (torch.randn(2), torch.zeros(2, dtype=torch.long))

    critic = MockCritic()

    ppo = StepwisePPO(
        env=env,
        policy=policy,
        critic=critic,
        batch_size=2
    )

    batch = TensorDict({"locs": torch.randn(2, 5, 2)}, batch_size=[2])

    # Run training step
    # We need to mock trainer and optimizers for manual optimization
    ppo.trainer = MagicMock()
    ppo.optimizers = MagicMock(return_value=torch.optim.Adam(ppo.parameters()))
    ppo.manual_backward = MagicMock()
    ppo.clip_gradients = MagicMock()

    loss = ppo.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert ppo.optimizers.called

if __name__ == "__main__":
    pytest.main([__file__])
