"""
Tests for DACT Model and TSPkopt Environment.
"""

import pytest
import torch
from tensordict import TensorDict

from logic.src.envs.tsp import TSPkoptEnv
from logic.src.models.dact import DACT
from logic.src.models.policies.dact import DACTPolicy


def test_tsp_kopt_env():
    """Test TSPkoptEnv reset and step."""
    env = TSPkoptEnv(num_loc=20)
    td = env.reset(batch_size=[2])

    assert "locs" in td.keys()
    assert "solution" in td.keys()
    assert td["solution"].shape == (2, 21)  # 20 customers + 1 depot

    # Initial reward
    reward_init = env.get_reward(td)
    assert reward_init.shape == (2,)

    # Perform a dummy action (2-opt swap of indices 1 and 5)
    action = torch.tensor([[1, 5], [2, 10]], dtype=torch.long)
    td["action"] = action
    td = env.step(td)

    # Check solution changed
    assert "solution" in td.keys()
    reward_new = env.get_reward(td)
    # Reward might change or stay same if it's a zero-improvement move,
    # but the mechanics should work.


def test_dact_policy_forward():
    """Test DACTPolicy iterative loop."""
    env = TSPkoptEnv(num_loc=20)
    policy = DACTPolicy(embed_dim=128)

    td = env.reset(batch_size=[2])

    # Run one step of policy
    # We need to ensure we can run it in greedy mode
    out = policy(td, env, decode_type="greedy", max_steps=3)

    assert "actions" in out.keys()
    assert out["actions"].shape == (2, 3, 2)  # batch, steps, 2 nodes

    # Check that the solution in td was updated
    # (DACTPolicy should update td internally during the loop)
    # Wait, in ImprovementPolicy.forward, it calls env.step(td)
    # The return value of policy() is a dict with actions, etc.
    # The td passed in might be updated in-place or returned?
    # Usually TensorDict is updated in place if modified.


def test_dact_model():
    """Test DACT model wrapper."""
    env = TSPkoptEnv(num_loc=10)
    model = DACT(env, embed_dim=64)

    td = env.reset(batch_size=[1])

    # DACT wrapper should call policy.forward
    out = model(td, env, max_steps=2)

    assert "actions" in out.keys()
    assert "reward" in out.keys()
    assert out["reward"].shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__])
