"""
Integration tests for training workflows.
"""


import pytest
import torch
from logic.src.envs import VRPPEnv, WCVRPEnv
from logic.src.models.policies import AttentionModelPolicy
from logic.src.pipeline.rl.core import PPO, REINFORCE
from pytorch_lightning import Trainer


class SimpleCritic(torch.nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.value = torch.nn.Linear(embed_dim, 1)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, td):
        # Return scalar value per batch
        batch_size = td.batch_size[0]
        return torch.randn(batch_size, device=td.device)


@pytest.mark.integration
def test_reinforce_training_loop():
    """Test REINFORCE training loop for a few steps."""
    # Initialize env with batch size matching training
    env = VRPPEnv(num_loc=10, batch_size=[2])
    policy = AttentionModelPolicy(
        env_name="vrpp",
        embed_dim=128,
        hidden_dim=128,
        n_encode_layers=2,
        n_heads=8,
    )

    # REINFORCE with rollout baseline
    module = REINFORCE(
        env=env,
        policy=policy,
        baseline="rollout",
        optimizer_kwargs={"lr": 1e-4},
        batch_size=2,
        train_data_size=10,
        val_data_size=10,
    )

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(module)
    assert trainer.global_step > 0


@pytest.mark.integration
def test_ppo_training_loop():
    """Test PPO training loop for a few steps."""
    env = WCVRPEnv(num_loc=10, batch_size=[1], check_env_specs=False)
    policy = AttentionModelPolicy(
        env_name="wcvrp",
        embed_dim=128,
        hidden_dim=128,
        n_encode_layers=2,
        n_heads=8,
    )
    critic = SimpleCritic(embed_dim=128)

    module = PPO(
        env=env,
        policy=policy,
        critic=critic,
        optimizer_kwargs={"lr": 1e-4},
        batch_size=1,
        train_data_size=10,
        val_data_size=10,
    )

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(module)
    assert trainer.global_step > 0
