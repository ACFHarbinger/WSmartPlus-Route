"""Tests for POMO and SymNCO RL algorithms."""

import pytest
import torch
from tensordict import TensorDict
from logic.src.pipeline.rl.core.pomo import POMO
from logic.src.pipeline.rl.core.symnco import SymNCO


class MockEnv:
    def reset(self, batch):
        return batch.clone()

    def get_num_starts(self, td):
        return 5

class MockPolicy(torch.nn.Module):
    def __init__(self, n_start=5, n_aug=1):
        super().__init__()
        self.n_start = n_start
        self.n_aug = n_aug

    def forward(self, td, env, strategy="sampling", num_starts=None):
        bs = td.batch_size[0]
        n_s = num_starts if num_starts is not None else self.n_start
        # reward: [batch * n_aug * n_s]
        # SymNCO passes td which might already be augmented
        # If n_aug > 1, the total batch size in td is bs * n_aug
        current_bs = td.batch_size[0]
        reward = torch.rand(current_bs * n_s)
        log_likelihood = torch.rand(current_bs * n_s)

        return {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "proj_embeddings": torch.rand(current_bs, 32)
        }

class TestPOMO:
    """Tests for the POMO algorithm."""

    @pytest.fixture
    def pomo_module(self):
        env = MockEnv()
        policy = MockPolicy()
        module = POMO(
            env=env,
            policy=policy,
            num_augment=1,
            num_starts=5,
            model_name="am"
        )
        # Mock lightning log method
        module.log = lambda *args, **kwargs: None
        return module

    def test_shared_step_train(self, pomo_module):
        """Test training step logic."""
        batch_size = 2
        batch = TensorDict({
            "locs": torch.rand(batch_size, 10, 2)
        }, batch_size=[batch_size])

        out = pomo_module.shared_step(batch, 0, "train")

        assert "loss" in out
        assert out["reward"].shape == (batch_size,)
        assert not torch.isnan(out["loss"])

    def test_shared_step_val(self, pomo_module):
        """Test validation step logic."""
        batch_size = 2
        batch = TensorDict({
            "locs": torch.rand(batch_size, 10, 2)
        }, batch_size=[batch_size])

        out = pomo_module.shared_step(batch, 0, "val")

        assert "loss" not in out
        assert out["reward"].shape == (batch_size,)

class TestSymNCO:
    """Tests for the SymNCO algorithm."""

    @pytest.fixture
    def symnco_module(self):
        env = MockEnv()
        policy = MockPolicy()
        module = SymNCO(
            env=env,
            policy=policy,
            num_augment=4,
            num_starts=5,
            alpha=0.2,
            beta=1.0,
            model_name="am"
        )
        # Mock lightning log method
        module.log = lambda *args, **kwargs: None
        return module

    def test_shared_step_train(self, symnco_module):
        """Test training step logic with losses."""
        batch_size = 2
        batch = TensorDict({
            "locs": torch.rand(batch_size, 10, 2)
        }, batch_size=[batch_size])

        # Need to mock augmentation if num_augment > 1
        class MockAug:
            def __call__(self, td):
                bs = td.batch_size[0]
                # Expand batch for 4 augmentations
                new_td = td.expand(4, bs).contiguous().view(bs * 4)
                return new_td

        symnco_module.augmentation = MockAug()
        symnco_module.num_augment = 4

        out = symnco_module.shared_step(batch, 0, "train")

        assert "loss" in out
        assert out["reward"].shape == (batch_size,)
        assert not torch.isnan(out["loss"])

    def test_shared_step_val(self, symnco_module):
        """Test validation step logic."""
        batch_size = 2
        batch = TensorDict({
            "locs": torch.rand(batch_size, 10, 2)
        }, batch_size=[batch_size])

        # Mock augmentation
        class MockAug:
            def __call__(self, td):
                bs = td.batch_size[0]
                return td.expand(4, bs).contiguous().view(bs * 4)

        symnco_module.augmentation = MockAug()

        out = symnco_module.shared_step(batch, 0, "val")

        assert out["reward"].shape == (batch_size,)
