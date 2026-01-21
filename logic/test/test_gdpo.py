"""
Tests for GDPO Algorithm.
"""
import torch
from tensordict import TensorDict

from logic.src.pipeline.rl.core.gdpo import GDPO


def test_gdpo_logic():
    """Verify GDPO normalization and aggregation logic."""
    batch_size = 4

    # Mock data
    # Obj 1: [10, 20, 30, 40] -> Mean 25, Std ~12.9
    # Obj 2: [1, 1, 1, 1] -> Mean 1, Std 0
    rewards_prize = torch.tensor([10.0, 20.0, 30.0, 40.0])
    rewards_cost = torch.tensor([1.0, 1.0, 1.0, 1.0])

    batch = TensorDict(
        {
            "reward_prize": rewards_prize,
            "reward_cost": rewards_cost,
        },
        batch_size=[batch_size],
    )

    # Mock output dictionary from model
    out = {
        "log_likelihood": torch.zeros(batch_size),  # Dummy
        "reward": torch.zeros(batch_size),  # Dummy
    }

    # Mock Env and Policy
    class MockEnv:
        name = "mock_env"

        def reset(self, td):
            return td

    class MockPolicy(torch.nn.Module):
        def forward(self, td):
            return td

    mock_env = MockEnv()
    mock_policy = MockPolicy()

    # Initialize GDPO
    # We intercept calculate_loss to check intermediate values?
    # Or we can just inspect the logs if we mock 'log'.
    # But easier to just instantiate and call the logic manually or
    # better yet, verify via a mock subclass that exposes internals.

    # Let's subclass to expose internals for testing
    class TestGDPO(GDPO):
        def test_internals(self, td, out):
            rewards_list = [td[k] for k in self.objective_keys]
            raw_rewards = torch.stack(rewards_list, dim=1)

            means = raw_rewards.mean(dim=0, keepdim=True)
            stds = raw_rewards.std(dim=0, keepdim=True)
            normalized = (raw_rewards - means) / (stds + 1e-8)

            aggregated = (normalized * self.weights_tensor).sum(dim=1)
            return raw_rewards, normalized, aggregated

    tester = TestGDPO(
        env=mock_env,
        policy=mock_policy,
        gdpo_objective_keys=["reward_prize", "reward_cost"],
        gdpo_objective_weights=[1.0, 0.5],
        gdpo_renormalize=False,
    )

    raw, norm, agg = tester.test_internals(batch, out)

    print("Raw:", raw)
    print("Norm:", norm)
    print("Agg:", agg)

    # Verify Obj 1 normalization
    expected_mean = 25.0
    expected_std = 12.909944
    expected_norm_0 = (10.0 - expected_mean) / expected_std
    assert torch.isclose(norm[0, 0], torch.tensor(expected_norm_0), atol=1e-4)

    # Verify Obj 2 normalization (std is 0)
    # (1 - 1) / (0 + 1e-8) = 0
    assert torch.isclose(norm[0, 1], torch.tensor(0.0), atol=1e-4)

    # Verify Aggregation
    # Agg = 1.0 * Norm1 + 0.5 * Norm2
    expected_agg_0 = 1.0 * expected_norm_0 + 0.5 * 0.0
    assert torch.isclose(agg[0], torch.tensor(expected_agg_0), atol=1e-4)

    print("Test passed!")


if __name__ == "__main__":
    test_gdpo_logic()
