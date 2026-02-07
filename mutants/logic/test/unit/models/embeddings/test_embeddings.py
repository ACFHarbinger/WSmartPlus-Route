"""Tests for embedding modules."""

import torch
import pytest
from tensordict import TensorDict
from logic.src.models.embeddings.context_embedding import VRPPContext
from logic.src.models.embeddings.dynamic_embedding import StaticEmbedding, DynamicEmbedding


class TestContextEmbedding:
    """Tests for ContextEmbeddings."""

    def test_vrpp_context(self):
        """Verify VRPPContext output shape."""
        batch = 2
        nodes = 5
        embed_dim = 16
        model = VRPPContext(embed_dim)

        # Mock embeddings: [batch, nodes, embed_dim]
        embeddings = torch.randn(batch, nodes, embed_dim)

        # Mock TensorDict
        td = TensorDict({
            "current_node": torch.zeros(batch, dtype=torch.long), # At depot/start
            "current_load": torch.rand(batch, 1)                  # Current load
        }, batch_size=batch)

        out = model(embeddings, td)
        # Output should be [batch, 1, embed_dim]
        assert out.shape == (batch, 1, embed_dim)

    def test_vrpp_context_no_load(self):
        """Verify fallback when load is missing (robustness)."""
        batch = 2
        nodes = 5
        embed_dim = 16
        model = VRPPContext(embed_dim)

        embeddings = torch.randn(batch, nodes, embed_dim)

        # Missing 'current_load'
        td = TensorDict({
            "current_node": torch.zeros(batch, dtype=torch.long)
        }, batch_size=batch)

        # Should not crash, returns embedding with zero context or similar
        out = model(embeddings, td)
        assert out.shape == (batch, 1, embed_dim)


class TestDynamicEmbedding:
    """Tests for DynamicEmbeddings."""

    def test_static_embedding(self):
        """Verify StaticEmbedding returns 0."""
        model = StaticEmbedding(16)
        td = TensorDict({}, batch_size=2)
        out = model(td)
        assert out == (0, 0, 0)
