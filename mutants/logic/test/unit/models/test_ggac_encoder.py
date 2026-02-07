"""Tests for GatedGraphAttConvEncoder."""

import pytest
import torch
from logic.src.models.subnets.encoders.ggac_encoder import GatedGraphAttConvEncoder, AttentionGatedConvolutionLayer


class TestGatedGraphAttConvEncoder:
    """Tests for the GatedGraphAttConvEncoder and its layers."""

    @pytest.fixture
    def layer_params(self):
        """Standard parameters for the layer."""
        return {
            "n_heads": 2,
            "embed_dim": 16,
            "feed_forward_hidden": 32,
            "normalization": "batch",
            "epsilon_alpha": 1e-5,
            "learn_affine": True,
            "track_stats": False,
            "mbeta": 0.1,
            "lr_k": 1.0,
            "n_groups": 3,
            "activation": "gelu",
            "af_param": 1.0,
            "threshold": 6.0,
            "replacement_value": 6.0,
            "n_params": 3,
            "uniform_range": [0.125, 0.33],
        }

    def test_layer_forward(self, layer_params):
        """Test forward pass of a single AttentionGatedConvolutionLayer."""
        layer = AttentionGatedConvolutionLayer(**layer_params)

        bs, n_nodes, dim = 2, 5, 16
        h = torch.rand(bs, n_nodes, dim)
        e = torch.rand(bs, n_nodes, n_nodes, dim)
        mask = torch.zeros(bs, n_nodes, n_nodes, dtype=torch.bool)

        h_out, e_out = layer(h, e, mask=mask)

        assert h_out.shape == (bs, n_nodes, dim)
        assert e_out.shape == (bs, n_nodes, n_nodes, dim)

    def test_encoder_initialization(self):
        """Test initialization of the full encoder."""
        encoder = GatedGraphAttConvEncoder(
            n_heads=2,
            embed_dim=16,
            n_layers=2,
            feed_forward_hidden=32
        )
        assert len(encoder.layers) == 2
        assert encoder.embed_dim == 16

    def test_encoder_forward_with_dist(self):
        """Test encoder forward pass with distance matrix."""
        bs, n_nodes, dim = 2, 5, 16
        encoder = GatedGraphAttConvEncoder(
            n_heads=4,
            embed_dim=dim,
            n_layers=2
        )

        x = torch.rand(bs, n_nodes, dim)
        dist = torch.rand(bs, n_nodes, n_nodes)

        out = encoder(x, dist=dist)

        assert out.shape == (bs, n_nodes, dim)
        assert not torch.isnan(out).any()

    def test_encoder_forward_no_dist(self):
        """Test encoder forward pass without distance matrix (fallback)."""
        bs, n_nodes, dim = 2, 5, 16
        encoder = GatedGraphAttConvEncoder(
            n_heads=2,
            embed_dim=dim,
            n_layers=1
        )

        x = torch.rand(bs, n_nodes, dim)
        out = encoder(x, dist=None)

        assert out.shape == (bs, n_nodes, dim)

    def test_encoder_forward_2d_dist(self):
        """Test encoder forward pass with a shared 2D distance matrix."""
        n_nodes, dim = 5, 16
        encoder = GatedGraphAttConvEncoder(
            n_heads=2,
            embed_dim=dim,
            n_layers=1
        )

        x = torch.rand(2, n_nodes, dim)
        dist = torch.rand(n_nodes, n_nodes)

        out = encoder(x, dist=dist)
        assert out.shape == (2, n_nodes, dim)
