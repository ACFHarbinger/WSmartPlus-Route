"""
Factory pattern for neural components (encoders, decoders).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn

from logic.src.models.subnets.decoders.deepaco import ACODecoder
from logic.src.models.subnets.decoders.gat import DeepGATDecoder
from logic.src.models.subnets.decoders.glimpse.decoder import GlimpseDecoder
from logic.src.models.subnets.decoders.mdam import MDAMDecoder
from logic.src.models.subnets.decoders.polynet import PolyNetDecoder
from logic.src.models.subnets.decoders.ptr import PointerDecoder
from logic.src.models.subnets.encoders.gac.encoder import GraphAttConvEncoder
from logic.src.models.subnets.encoders.gat.encoder import GraphAttentionEncoder
from logic.src.models.subnets.encoders.gcn.encoder import GraphConvolutionEncoder
from logic.src.models.subnets.encoders.ggac.encoder import GatedGraphAttConvEncoder
from logic.src.models.subnets.encoders.mdam.encoder import MDAMGraphAttentionEncoder
from logic.src.models.subnets.encoders.mlp.encoder import MLPEncoder
from logic.src.models.subnets.encoders.moe.encoder import MoEGraphAttentionEncoder
from logic.src.models.subnets.encoders.nargnn import NARGNNEncoder
from logic.src.models.subnets.encoders.tgc.encoder import TransGraphConvEncoder


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
    elif decoder_type == "mdam":
        return MDAMDecoder(**kwargs)
    elif decoder_type == "polynet":
        return PolyNetDecoder(**kwargs)
    elif decoder_type == "aco":
        return ACODecoder(**kwargs)
    else:
        raise ValueError(
            f"Unknown decoder_type: {decoder_type}. Choose from 'attention', 'deep', 'pointer', 'mdam', 'polynet', 'aco'."
        )


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


class NARComponentFactory(NeuralComponentFactory):
    """Factory for Non-Autoregressive Models (DeepACO, GFACS, NARGNN)."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create NARGNN Encoder."""
        return NARGNNEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "aco", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)


class MDAMComponentFactory(NeuralComponentFactory):
    """Factory for MDAM Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create MDAM Graph Attention Encoder."""
        return MDAMGraphAttentionEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "mdam", **kwargs: Any) -> nn.Module:
        """Create decoder based on decoder_type."""
        return _create_decoder_by_type(decoder_type, **kwargs)


class GFACSComponentFactory(NeuralComponentFactory):
    """Factory for GFACS Models."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create GFACS Encoder."""
        from logic.src.models.subnets.encoders.gfacs.encoder import GFACSEncoder

        return GFACSEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "aco", **kwargs: Any) -> nn.Module:
        """Create ACO Decoder."""
        # Reuse ACODecoder from DeepACO
        from logic.src.models.subnets.decoders.deepaco import ACODecoder

        return ACODecoder(**kwargs)
