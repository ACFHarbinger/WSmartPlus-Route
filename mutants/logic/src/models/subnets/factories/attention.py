"""Attention model component factory."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class AttentionComponentFactory(NeuralComponentFactory):
    """Factory for Angle-based Attention Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create Graph Attention Encoder."""
        return GraphAttentionEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)
