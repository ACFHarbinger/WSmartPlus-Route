"""Deep learning models for Vehicle Routing and Scheduling Problems.

This package provides a comprehensive set of neural architectures for combinatorial
optimization, organized into constructive, iterative improvement, and transductive
modeling paradigms.

Attributes:
    AttentionModel: The standard Encoder-Decoder with Multi-Head Attention.
    TemporalAttentionModel: Attention model with time-window awareness.
    CriticNetwork: Value function estimator for RL training.
    MandatoryManager: HRL manager for temporal decision-making.
    WeightAdjustmentRNN: Meta-learning RNN for dynamic weight scaling.
    HyperNetwork: Parameters generator for task-conditioned adapters.
    MatNet: Matrix Encoding Networks for asymmetric/stochastic problems.
    MDAM: Multi-Decoder Attention Model for diverse solution sampling.
    DeepACO: Neural-guided Ant Colony Optimization.
    DACT: Dual-Aspect Collaborative Transformer for local search.
    N2S: Neural Neighborhood Search for iterative improvement.
    NeuOpt: Neural Optimizer for routing refinement.
    PointerNetwork: RNN-based pointer architecture (Vinyals et al.).
    ActiveSearch: Online model adaptation via test-time optimization.
    EAS: Efficient Active Search via adapter tuning.

Example:
    >>> from logic.src.models import AttentionModel
    >>> model = AttentionModel(env_name="vrp", num_loc=50)
    >>> td = env.reset()
    >>> out = model(td, decode_type="greedy")
"""

from . import common as common
from . import core as core
from . import meta as meta
from . import subnets as subnets
from .common.critic_network.policy import CriticNetwork as CriticNetwork
from .common.transductive.active_search import ActiveSearch as ActiveSearch
from .common.transductive.base import TransductiveModel as TransductiveModel
from .common.transductive.eas import EAS as EAS
from .core.attention_model import AttentionModel as AttentionModel
from .core.dact import DACT as DACT
from .core.deepaco import DeepACO as DeepACO
from .core.gfacs import GFACS as GFACS
from .core.glop import GLOP as GLOP
from .core.matnet import MatNet as MatNet
from .core.mdam import MDAM as MDAM
from .core.moe import (
    MoEAttentionModel as MoEAttentionModel,
)
from .core.moe import (
    MoETemporalAttentionModel as MoETemporalAttentionModel,
)
from .core.n2s import N2S as N2S
from .core.nargnn import NARGNN as NARGNN
from .core.neuopt import NeuOpt as NeuOpt
from .core.pointer_network import PointerNetwork as PointerNetwork
from .core.polynet import PolyNet as PolyNet
from .core.temporal_attention_model import TemporalAttentionModel as TemporalAttentionModel
from .meta.hrl_manager import MandatoryManager as MandatoryManager
from .meta.hypernet import (
    HyperNetwork as HyperNetwork,
)
from .meta.hypernet import (
    HyperNetworkOptimizer as HyperNetworkOptimizer,
)
from .meta.weight_adjustment_rnn import WeightAdjustmentRNN as WeightAdjustmentRNN
from .subnets import (
    GatedGraphAttConvEncoder as GatedGraphAttConvEncoder,
)
from .subnets import (
    GatedRecurrentUnitFillPredictor as GatedRecurrentUnitFillPredictor,
)
from .subnets import (
    GraphAttConvEncoder as GraphAttConvEncoder,
)
from .subnets import (
    GraphAttentionEncoder as GraphAttentionEncoder,
)
from .subnets import (
    TransGraphConvEncoder as TransGraphConvEncoder,
)

__all__ = [
    # Core Models
    "AttentionModel",
    "TemporalAttentionModel",
    "CriticNetwork",
    "MandatoryManager",
    "WeightAdjustmentRNN",
    "HyperNetwork",
    "HyperNetworkOptimizer",
    "MatNet",
    "MDAM",
    "PolyNet",
    "GLOP",
    "DeepACO",
    "GFACS",
    "NARGNN",
    "DACT",
    "N2S",
    "NeuOpt",
    "MoEAttentionModel",
    "MoETemporalAttentionModel",
    "PointerNetwork",
    "TransductiveModel",
    "ActiveSearch",
    "EAS",
    # Encoders
    "GraphAttentionEncoder",
    "GraphAttConvEncoder",
    "GatedGraphAttConvEncoder",
    "TransGraphConvEncoder",
    # Predictors
    "GatedRecurrentUnitFillPredictor",
]
