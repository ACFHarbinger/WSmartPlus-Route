"""MLP component factory."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from logic.src.models.subnets.encoders.mlp.encoder import MLPEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class MLPComponentFactory(NeuralComponentFactory):
    """Factory for MLP-based Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create MLP Encoder."""
        return MLPEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)
