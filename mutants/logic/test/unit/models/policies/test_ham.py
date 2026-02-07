
import pytest
import torch
from tensordict import TensorDict

from logic.src.models.policies.ham import HAMPolicy
from logic.src.models.subnets.encoders.ham_encoder import HAMEncoder
from logic.src.models.modules.ham_attention import HeterogeneousAttentionLayer
from logic.src.models.embeddings.pdp import PDPInitEmbedding

class TestHAM:
    @pytest.fixture
    def setup_data(self):
        batch_size = 2
        num_loc = 5 # 5 pairs -> 10 nodes
        embed_dim = 16

        locs = torch.randn(batch_size, 2 * num_loc, 2)
        depot = torch.randn(batch_size, 2)

        return TensorDict({
            "locs": locs,
            "depot": depot
        }, batch_size=[batch_size])

    def test_het_attn_layer(self):
        node_types = ["a", "b"]
        edge_types = [("a", "to", "b"), ("b", "to", "a"), ("a", "to", "a")]
        layer = HeterogeneousAttentionLayer(
            node_types=node_types,
            edge_types=edge_types,
            embed_dim=16,
            num_heads=2
        )

        x_dict = {
            "a": torch.randn(2, 5, 16),
            "b": torch.randn(2, 3, 16)
        }

        out = layer(x_dict)
        assert out["a"].shape == (2, 5, 16)
        assert out["b"].shape == (2, 3, 16)

    def test_ham_encoder(self, setup_data):
        # 1. Init Embedding
        init_embed = PDPInitEmbedding(embed_dim=16)
        embeddings = init_embed(setup_data)

        encoder = HAMEncoder(
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            normalization="layer" # Use layer norm to avoid singlet dimension issues with batch=2
        )

        res_embeddings, init_res = encoder(embeddings)

        # Expected shape: (batch, 1 + 2*N, embed_dim)
        assert res_embeddings.shape == (2, 11, 16)
        assert init_res.shape == (2, 11, 16)

    def test_ham_policy(self, setup_data):
        from logic.src.envs.pdp import PDPEnv
        env = PDPEnv(num_loc=5)

        policy = HAMPolicy(
            embed_dim=16,
            num_encoder_layers=1,
            num_heads=2,
            env_name="pdp",
            normalization="layer"
        )

        # Test forward (acting)
        policy.eval()

        # Reset env to get proper state if needed, though setup_data attempts to mock it
        # Better to let env reset for consistency
        td = env.reset(batch_size=[2])

        out = policy(td, env, decode_type="greedy")

        assert "actions" in out
        assert out["actions"].shape[0] == 2
