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
