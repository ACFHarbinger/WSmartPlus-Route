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
    @abstractmethod
    def create_encoder(self, **kwargs) -> nn.Module:
        pass

    @abstractmethod
    def create_decoder(self, **kwargs) -> nn.Module:
        pass

class AttentionComponentFactory(NeuralComponentFactory):
    def create_encoder(self, **kwargs) -> nn.Module:
        return GraphAttentionEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        return AttentionDecoder(**kwargs)

class GCNComponentFactory(NeuralComponentFactory):
    def create_encoder(self, **kwargs) -> nn.Module:
        return GraphConvolutionEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        return AttentionDecoder(**kwargs)

class GACComponentFactory(NeuralComponentFactory):
    def create_encoder(self, **kwargs) -> nn.Module:
        return GraphAttConvEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        return AttentionDecoder(**kwargs)

class TGCComponentFactory(NeuralComponentFactory):
    def create_encoder(self, **kwargs) -> nn.Module:
        return TransGraphConvEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        return AttentionDecoder(**kwargs)

class GGACComponentFactory(NeuralComponentFactory):
    def create_encoder(self, **kwargs) -> nn.Module:
        return GatedGraphAttConvEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        return AttentionDecoder(**kwargs)

class MLPComponentFactory(NeuralComponentFactory):
    def create_encoder(self, **kwargs) -> nn.Module:
        return MLPEncoder(**kwargs)

    def create_decoder(self, **kwargs) -> nn.Module:
        return AttentionDecoder(**kwargs)
