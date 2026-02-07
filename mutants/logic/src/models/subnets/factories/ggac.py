"""GGAC component factory."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from logic.src.models.subnets.encoders.ggac.encoder import GatedGraphAttConvEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class GGACComponentFactory(NeuralComponentFactory):
    """Factory for Gated Graph Attention Convolution Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create Gated Graph Attention Convolution Encoder."""
        return GatedGraphAttConvEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)
