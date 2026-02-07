
import pytest
import torch
from logic.src.envs.jssp import JSSPEnv
from logic.src.models.policies.l2d import L2DPolicy
from logic.src.models.subnets.encoders.l2d.encoder import L2DEncoder

class TestL2D:
    @pytest.fixture
    def env(self):
        return JSSPEnv(num_jobs=3, num_machines=3)

    def test_l2d_encoder(self, env):
        td = env.reset(batch_size=[2])
        encoder = L2DEncoder(embed_dim=16, num_layers=2)

        job_embeddings, full_embeddings = encoder(td)

        # Expect (B, J, D)
        assert job_embeddings.shape == (2, 3, 16)
        # Expect (B, J*M, D)
        assert full_embeddings.shape == (2, 9, 16)

    def test_l2d_policy(self, env):
        policy = L2DPolicy(
            embed_dim=16,
            num_encoder_layers=1,
            env_name="jssp"
        )

        # Reset env to get fresh state (policy resets internally too but good to have consistent td)
        td = env.reset(batch_size=[2])

        # Test full rollout (forward)
        out = policy(td, env, decode_type="greedy")

        assert "reward" in out
        assert "actions" in out
        # Reward shape: (B,)
        assert out["reward"].shape == (2,)
        # Actions shape: (B, Steps) -> J*M steps
        assert out["actions"].shape[1] == 9
