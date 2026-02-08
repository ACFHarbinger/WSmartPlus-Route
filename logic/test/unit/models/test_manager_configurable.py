
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from logic.src.models.hrl_manager.model import GATLSTManager
from logic.src.models.subnets.factories.attention import AttentionComponentFactory
from logic.src.models.subnets.factories.gcn import GCNComponentFactory
from logic.src.models.hrl_manager.temporal_encoder import TemporalEncoder

class TestManagerConfigurableEncoders:

    def test_init_temporal_gru(self):
        """Test initializing manager with GRU temporal encoder."""
        manager = GATLSTManager(temporal_encoder_type="gru")
        assert isinstance(manager.temporal_encoder, TemporalEncoder)
        assert manager.temporal_encoder.rnn_type == "gru"
        assert isinstance(manager.temporal_encoder.rnn, nn.GRU)

    def test_init_temporal_lstm(self):
        """Test initializing manager with LSTM temporal encoder (default)."""
        manager = GATLSTManager(temporal_encoder_type="lstm")
        assert isinstance(manager.temporal_encoder, TemporalEncoder)
        assert manager.temporal_encoder.rnn_type == "lstm"
        assert isinstance(manager.temporal_encoder.rnn, nn.LSTM)

    def test_init_with_attention_factory(self):
        """Test initializing manager with AttentionComponentFactory."""
        factory = AttentionComponentFactory()
        manager = GATLSTManager(component_factory=factory)

        # Should create GraphAttentionEncoder
        from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder
        assert isinstance(manager.gat_encoder, GraphAttentionEncoder)

    def test_init_with_gcn_factory(self):
        """Test initializing manager with GCNComponentFactory."""
        factory = GCNComponentFactory()
        manager = GATLSTManager(component_factory=factory)

        # Should create GraphConvolutionEncoder
        from logic.src.models.subnets.encoders.gcn import GraphConvolutionEncoder
        assert isinstance(manager.gat_encoder, GraphConvolutionEncoder)
