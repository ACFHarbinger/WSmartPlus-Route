"""
This package contains deep learning models for solving Vehicle Routing Problems.
"""
from .critic_network import CriticNetwork as CriticNetwork
from logic.src.pipeline.reinforcement_learning.core.reinforce_baselines import NoBaseline as NoBaseline, ExponentialBaseline as ExponentialBaseline, CriticBaseline as CriticBaseline, RolloutBaseline as RolloutBaseline, WarmupBaseline as WarmupBaseline, POMOBaseline as POMOBaseline

from .attention_model import AttentionModel as AttentionModel
from .temporal_am import TemporalAttentionModel as TemporalAttentionModel
from .deep_decoder_am import DeepDecoderAttentionModel as DeepDecoderAttentionModel
from .subnets import GraphAttentionEncoder as GraphAttentionEncoder, GraphAttConvEncoder as GraphAttConvEncoder, TransGraphConvEncoder as TransGraphConvEncoder, GatedRecurrentFillPredictor as GatedRecurrentFillPredictor, GatedGraphAttConvEncoder as GatedGraphAttConvEncoder

from .gat_lstm_manager import GATLSTManager as GATLSTManager

from .hypernet import Hypernetwork as Hypernetwork, HypernetworkOptimizer as HypernetworkOptimizer
from .meta_rnn import WeightAdjustmentRNN as WeightAdjustmentRNN