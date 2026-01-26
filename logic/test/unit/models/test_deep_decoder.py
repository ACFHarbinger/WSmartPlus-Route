"""Tests for DeepDecoder."""

import torch
import torch.nn as nn
from logic.src.models.subnets.deep_decoder import DeepAttentionModelFixed, DeepDecoder


def test_deep_decoder_init():
    """Verify initialization of DeepDecoder."""
    model = DeepDecoder(embed_dim=16, hidden_dim=16, n_heads=2, n_layers=1)
    assert isinstance(model.decoder, nn.Module)
    assert model.embed_dim == 16


def test_deep_decoder_precompute():
    """Verify precomputation with different aggregation strategies."""
    model = DeepDecoder(embed_dim=16, hidden_dim=16, n_heads=2, n_layers=1, aggregation_graph="avg")

    batch, nodes, dim = 2, 5, 16
    embeddings = torch.randn(batch, nodes, dim)

    # Test avg
    fixed_avg = model._precompute(embeddings)
    assert fixed_avg.node_embeddings.shape == (batch, nodes, dim)
    assert fixed_avg.context_node_projected.shape == (batch, 1, dim)

    # Test sum
    model.aggregation_graph = "sum"
    fixed_sum = model._precompute(embeddings)
    assert fixed_sum.context_node_projected.shape == (batch, 1, dim)

    # Test max
    model.aggregation_graph = "max"
    fixed_max = model._precompute(embeddings)
    assert fixed_max.context_node_projected.shape == (batch, 1, dim)


def test_deep_decoder_get_log_p():
    """Verify log probability computation."""
    model = DeepDecoder(embed_dim=16, hidden_dim=16, n_heads=2, n_layers=1)

    batch, nodes, dim = 2, 5, 16
    embeddings = torch.randn(batch, nodes, dim)
    fixed = model._precompute(embeddings)

    from unittest.mock import MagicMock

    state = MagicMock()
    # Mock current node (1 per batch)
    state.get_current_node.return_value = torch.zeros(batch, 1, dtype=torch.long)
    # Mock mask
    state.get_mask.return_value = torch.zeros(batch, nodes, dtype=torch.bool)

    log_p, mask = model._get_log_p(fixed, state)

    assert log_p.shape == (batch, 1, nodes)
    assert mask.shape == (batch, nodes)


def test_deep_decoder_slicing():
    """Verify slicing of fixed context."""
    node_embeddings = torch.randn(4, 5, 16)
    context_node_projected = torch.randn(4, 1, 16)
    fixed = DeepAttentionModelFixed(node_embeddings, context_node_projected)

    sliced = fixed[0:2]
    assert sliced.node_embeddings.shape == (2, 5, 16)
    assert sliced.context_node_projected.shape == (2, 1, 16)
