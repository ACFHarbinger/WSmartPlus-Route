"""
This module contains the Factory Pattern implementation for creating neural model components.
"""
from abc import ABC, abstractmethod
import torch.nn as nn

from logic.src.models.subnets.gat_encoder import GraphAttentionEncoder
from logic.src.models.subnets.gcn_encoder import GraphConvolutionEncoder
from logic.src.models.subnets.mlp_encoder import MLPEncoder
from logic.src.models.subnets.gac_encoder import GraphAttConvEncoder
from logic.src.models.subnets.tgc_encoder import TransGraphConvEncoder
from logic.src.models.subnets.ggac_encoder import GatedGraphAttConvEncoder
from logic.src.models.subnets.attention_decoder import AttentionDecoder

class NeuralComponentFactory(ABC):
    """
    Abstract Factory for creating neural components (Encoders and Decoders).
    """
    @abstractmethod
    def create_encoder(self, **kwargs) -> nn.Module:
        """Create an encoder instance."""
        pass

    @abstractmethod
    def create_decoder(self, **kwargs) -> nn.Module:
        """Create a decoder instance."""
        pass

class AttentionComponentFactory(NeuralComponentFactory):
    """Factory for Angle-based Attention Models."""
    def create_encoder(self, **kwargs) -> nn.Module:
        """Create Graph Attention Encoder."""
        return GraphAttentionEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        """Create Attention Decoder."""
        return AttentionDecoder(**kwargs)

class GCNComponentFactory(NeuralComponentFactory):
    """Factory for Graph Convolutional Network Models."""
    def create_encoder(self, **kwargs) -> nn.Module:
        """Create GCN Encoder."""
        return GraphConvolutionEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        """Create Attention Decoder."""
        return AttentionDecoder(**kwargs)

class GACComponentFactory(NeuralComponentFactory):
    """Factory for Graph Attention Convolution Models."""
    def create_encoder(self, **kwargs) -> nn.Module:
        """Create Graph Attention Convolution Encoder."""
        return GraphAttConvEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        """Create Attention Decoder."""
        return AttentionDecoder(**kwargs)

class TGCComponentFactory(NeuralComponentFactory):
    """Factory for Transformer Graph Convolution Models."""
    def create_encoder(self, **kwargs) -> nn.Module:
        """Create Transformer Graph Convolution Encoder."""
        return TransGraphConvEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        """Create Attention Decoder."""
        return AttentionDecoder(**kwargs)

class GGACComponentFactory(NeuralComponentFactory):
    """Factory for Gated Graph Attention Convolution Models."""
    def create_encoder(self, **kwargs) -> nn.Module:
        """Create Gated Graph Attention Convolution Encoder."""
        return GatedGraphAttConvEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        """Create Attention Decoder."""
        return AttentionDecoder(**kwargs)

class MLPComponentFactory(NeuralComponentFactory):
    """Factory for MLP-based Models."""
    def create_encoder(self, **kwargs) -> nn.Module:
        """Create MLP Encoder."""
        return MLPEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        """Create Attention Decoder."""
        return AttentionDecoder(**kwargs)

class MoEComponentFactory(NeuralComponentFactory):
    """Factory for Mixture of Experts Models."""
    def __init__(self, num_experts=4, k=2, noisy_gating=True):
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

    def create_encoder(self, **kwargs) -> nn.Module:
        """Create MoE Graph Attention Encoder."""
        from logic.src.models.subnets.moe_encoder import MoEGraphAttentionEncoder
        # Inject MoE params
        kwargs['num_experts'] = self.num_experts
        kwargs['k'] = self.k
        kwargs['noisy_gating'] = self.noisy_gating
        return MoEGraphAttentionEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        """Create Attention Decoder."""
        return AttentionDecoder(**kwargs)
