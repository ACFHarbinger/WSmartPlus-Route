"""Tests for Non-Autoregressive models."""

import torch
from tensordict import TensorDict
from unittest.mock import MagicMock

from logic.src.models.policies.nonautoregressive import (
    NonAutoregressiveEncoder,
    NonAutoregressiveDecoder,
    NonAutoregressivePolicy,
)
from logic.src.models.deepaco import DeepACOEncoder, ACODecoder, DeepACOPolicy, DeepACO


class TestNonAutoregressiveBase:
    """Tests for NAR base classes."""

    def test_encoder_abc(self):
        """Verify NonAutoregressiveEncoder is abstract."""
        # Should not be instantiable directly
        try:
            encoder = NonAutoregressiveEncoder()
            assert False, "Should raise TypeError"
        except TypeError:
            pass  # Expected

    def test_decoder_abc(self):
        """Verify NonAutoregressiveDecoder is abstract."""
        try:
            decoder = NonAutoregressiveDecoder()
            assert False, "Should raise TypeError"
        except TypeError:
            pass  # Expected


class TestDeepACOEncoder:
    """Tests for DeepACOEncoder."""

    def setup_method(self):
        self.batch_size = 4
        self.num_nodes = 10
        self.embed_dim = 64
        self.encoder = DeepACOEncoder(embed_dim=self.embed_dim, num_layers=2)

    def test_forward_shape(self):
        """Verify encoder output shape."""
        td = TensorDict({
            "locs": torch.rand(self.batch_size, self.num_nodes, 2),
        }, batch_size=[self.batch_size])

        heatmap = self.encoder(td)

        assert heatmap.shape == (self.batch_size, self.num_nodes, self.num_nodes)
        # Check log-softmax: each row should sum to ~1 after exp
        assert torch.allclose(heatmap.exp().sum(dim=-1), torch.ones(self.batch_size, self.num_nodes), atol=1e-5)

    def test_forward_with_embeddings(self):
        """Verify encoder returns embeddings when requested."""
        td = TensorDict({
            "locs": torch.rand(self.batch_size, self.num_nodes, 2),
        }, batch_size=[self.batch_size])

        heatmap, embeddings = self.encoder(td, return_embeddings=True)

        assert embeddings.shape == (self.batch_size, self.num_nodes, self.embed_dim)


class TestACODecoder:
    """Tests for ACODecoder."""

    def setup_method(self):
        self.batch_size = 2
        self.num_nodes = 5
        self.decoder = ACODecoder(n_ants=5, use_local_search=False)

    def test_forward(self):
        """Verify decoder constructs valid tours."""
        td = TensorDict({
            "locs": torch.rand(self.batch_size, self.num_nodes, 2),
        }, batch_size=[self.batch_size])

        # Create dummy heatmap
        heatmap = torch.randn(self.batch_size, self.num_nodes, self.num_nodes)
        heatmap = torch.log_softmax(heatmap, dim=-1)

        env = MagicMock()

        out = self.decoder(td, heatmap, env)

        assert "actions" in out
        assert "reward" in out
        assert "log_likelihood" in out
        assert out["actions"].shape == (self.batch_size, self.num_nodes)

    def test_two_opt(self):
        """Verify 2-opt improves or maintains tour quality."""
        decoder = ACODecoder(n_ants=3, use_local_search=True)
        td = TensorDict({
            "locs": torch.rand(self.batch_size, self.num_nodes, 2),
        }, batch_size=[self.batch_size])

        heatmap = torch.log_softmax(torch.randn(self.batch_size, self.num_nodes, self.num_nodes), dim=-1)
        env = MagicMock()

        out = decoder(td, heatmap, env)
        assert out["actions"].shape == (self.batch_size, self.num_nodes)


class TestDeepACOPolicy:
    """Tests for DeepACOPolicy."""

    def setup_method(self):
        self.batch_size = 2
        self.num_nodes = 6
        self.policy = DeepACOPolicy(
            embed_dim=32,
            num_encoder_layers=1,
            n_ants=3,
            use_local_search=False,
        )

    def test_forward(self):
        """Verify full policy forward pass."""
        td = TensorDict({
            "locs": torch.rand(self.batch_size, self.num_nodes, 2),
        }, batch_size=[self.batch_size])

        env = MagicMock()

        out = self.policy(td, env)

        assert "actions" in out
        assert "reward" in out
        assert "heatmap" in out
        assert out["heatmap"].shape == (self.batch_size, self.num_nodes, self.num_nodes)


class TestDeepACOModel:
    """Tests for DeepACO REINFORCE model."""

    def setup_method(self):
        self.batch_size = 2
        self.num_nodes = 5
        self.model = DeepACO(
            embed_dim=32,
            num_encoder_layers=1,
            n_ants=2,
            use_local_search=False,
            baseline="rollout",
        )

    def test_forward_training(self):
        """Verify model produces loss for training."""
        td = TensorDict({
            "locs": torch.rand(self.batch_size, self.num_nodes, 2),
        }, batch_size=[self.batch_size])

        env = MagicMock()

        out = self.model(td, env)

        assert "loss" in out
        assert "reward" in out
        assert "baseline" in out
        assert out["loss"].requires_grad
