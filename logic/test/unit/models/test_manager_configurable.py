
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from logic.src.models.meta.hrl_manager import MandatoryManager
from logic.src.models.meta.hrl_manager.temporal_encoder import TemporalEncoder
from logic.src.models.subnets.factories.attention import AttentionComponentFactory
from logic.src.models.subnets.factories.gcn import GCNComponentFactory

class TestManagerConfigurableEncoders:

    def test_init_temporal_gru(self):
        """Test initializing manager with GRU temporal encoder."""
        manager = MandatoryManager(temporal_encoder_type="gru")
        assert isinstance(manager.temporal_encoder, TemporalEncoder)
        assert manager.temporal_encoder.rnn_type == "gru"
        assert isinstance(manager.temporal_encoder.rnn, nn.GRU)

    def test_init_temporal_lstm(self):
        """Test initializing manager with LSTM temporal encoder (default)."""
        manager = MandatoryManager(temporal_encoder_type="lstm")
        assert isinstance(manager.temporal_encoder, TemporalEncoder)
        assert manager.temporal_encoder.rnn_type == "lstm"
        assert isinstance(manager.temporal_encoder.rnn, nn.LSTM)

    def test_init_with_attention_factory(self):
        """Test initializing manager with AttentionComponentFactory."""
        factory = AttentionComponentFactory()
        manager = MandatoryManager(component_factory=factory)

        # Should create GraphAttentionEncoder
        from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder
        assert isinstance(manager.gat_encoder, GraphAttentionEncoder)

    def test_init_with_gcn_factory(self):
        """Test initializing manager with GCNComponentFactory."""
        factory = GCNComponentFactory()
        manager = MandatoryManager(component_factory=factory)

        # Should create GraphConvolutionEncoder
        from logic.src.models.subnets.encoders.gcn import GraphConvolutionEncoder
        assert isinstance(manager.gat_encoder, GraphConvolutionEncoder)
