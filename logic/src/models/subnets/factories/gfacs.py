"""GFACS component factory."""

from __future__ import annotations

from typing import Any

from torch import nn

from .base import NeuralComponentFactory


class GFACSComponentFactory(NeuralComponentFactory):
    """Factory for GFACS Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create GFACS Encoder."""
        from logic.src.models.subnets.encoders.gfacs.encoder import GFACSEncoder

        return GFACSEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "aco", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create ACO Decoder."""
        # Reuse ACODecoder from DeepACO
        from logic.src.models.subnets.decoders.deepaco import ACODecoder

        return ACODecoder(**kwargs)
