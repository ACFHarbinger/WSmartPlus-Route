"""
Factory pattern for neural components (encoders, decoders).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
from logic.src.models.subnets.decoders.gat_decoder import DeepGATDecoder
from logic.src.models.subnets.decoders.glimpse_decoder import GlimpseDecoder
from logic.src.models.subnets.decoders.ptr_decoder import PointerDecoder
from logic.src.models.subnets.encoders.gac_encoder import GraphAttConvEncoder
from logic.src.models.subnets.encoders.gat_encoder import GraphAttentionEncoder
from logic.src.models.subnets.encoders.gcn_encoder import GraphConvolutionEncoder
from logic.src.models.subnets.encoders.ggac_encoder import GatedGraphAttConvEncoder
from logic.src.models.subnets.encoders.mlp_encoder import MLPEncoder
from logic.src.models.subnets.encoders.moe_encoder import MoEGraphAttentionEncoder
from logic.src.models.subnets.encoders.tgc_encoder import TransGraphConvEncoder


def _create_decoder_by_type(decoder_type: str, **kwargs: Any) -> nn.Module:
    """
    Create a decoder instance based on decoder_type.

    Args:
        decoder_type: Type of decoder ('attention', 'deep', 'pointer').
        **kwargs: Arguments to pass to the decoder constructor.

    Returns:
        nn.Module: The decoder instance.

    Raises:
        ValueError: If decoder_type is not recognized.
    """
    decoder_type = decoder_type.lower()
    if decoder_type == "attention":
        return GlimpseDecoder(**kwargs)
    elif decoder_type in ("deep", "gat", "graph_attention"):
        return DeepGATDecoder(**kwargs)
    elif decoder_type == "pointer":
        return PointerDecoder(**kwargs)
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}. Choose from 'attention', 'deep', 'pointer'.")


class NeuralComponentFactory(ABC):
    """
    Abstract Factory for creating neural components (Encoders and Decoders).
    """

    @abstractmethod
    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create an encoder instance."""
        pass

    @abstractmethod
    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:
        """Create a decoder instance based on decoder_type."""
        pass


class AttentionComponentFactory(NeuralComponentFactory):
    """Factory for Angle-based Attention Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create Graph Attention Encoder."""
        return GraphAttentionEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)


class GCNComponentFactory(NeuralComponentFactory):
    """Factory for Graph Convolutional Network Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create GCN Encoder."""
        return GraphConvolutionEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)


class GACComponentFactory(NeuralComponentFactory):
    """Factory for Graph Attention Convolution Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create Graph Attention Convolution Encoder."""
        return GraphAttConvEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)


class TGCComponentFactory(NeuralComponentFactory):
    """Factory for Transformer Graph Convolution Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create Transformer Graph Convolution Encoder."""
        return TransGraphConvEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)


class GGACComponentFactory(NeuralComponentFactory):
    """Factory for Gated Graph Attention Convolution Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create Gated Graph Attention Convolution Encoder."""
        return GatedGraphAttConvEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)


class MLPComponentFactory(NeuralComponentFactory):
    """Factory for MLP-based Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create MLP Encoder."""
        return MLPEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)


class MoEComponentFactory(NeuralComponentFactory):
    """Factory for Mixture of Experts Models."""

    def __init__(self, num_experts: int = 4, k: int = 2, noisy_gating: bool = True) -> None:
        """
        Initialize the MoE Component Factory.

        Args:
            num_experts: Number of experts in the mixture.
            k: Number of experts to select per token.
            noisy_gating: Whether to add noise to the gating mechanism.
        """
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create MoE Graph Attention Encoder."""

        # Inject MoE params
        kwargs["num_experts"] = self.num_experts
        kwargs["k"] = self.k
        kwargs["noisy_gating"] = self.noisy_gating
        return MoEGraphAttentionEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)
