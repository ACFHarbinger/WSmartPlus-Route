
import pytest
import torch
from logic.src.models.subnets.modules.pointer_attn_moe import PointerAttnMoE

class TestPointerAttnMoE:
    @pytest.fixture
    def setup_data(self):
        batch_size = 2
        graph_size = 10
        num_steps = 3
        embed_dim = 16
        num_heads = 4

        query = torch.randn(batch_size, num_steps, embed_dim)
        key = torch.randn(batch_size, graph_size, embed_dim)
        value = torch.randn(batch_size, graph_size, embed_dim)
        logit_key = torch.randn(batch_size, graph_size, embed_dim)

        return query, key, value, logit_key, batch_size, graph_size, num_steps, embed_dim, num_heads

    def test_forward(self, setup_data):
        query, key, value, logit_key, batch_size, graph_size, num_steps, embed_dim, num_heads = setup_data

        module = PointerAttnMoE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_experts=4,
            k=2
        )

        logits = module(query, key, value, logit_key)

        assert logits.shape == (batch_size, num_steps, graph_size)
        assert not torch.isnan(logits).any()

    def test_masking(self, setup_data):
        query, key, value, logit_key, batch_size, graph_size, num_steps, embed_dim, num_heads = setup_data

        module = PointerAttnMoE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mask_inner=True
        )

        # Create a mask (batch, graph_size)
        mask = torch.zeros(batch_size, graph_size, dtype=torch.bool)
        mask[:, :5] = True # Mask first 5 nodes

        logits = module(query, key, value, logit_key, attn_mask=mask)

        assert logits.shape == (batch_size, num_steps, graph_size)
