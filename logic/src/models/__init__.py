"""
This package contains deep learning models for solving Vehicle Routing and Scheduling Problems.

It includes a variety of neural architectures:

Constructive Models (Autoregressive):
-   `AttentionModel`: The standard Encoder-Decoder with Multi-Head Attention (Kool et al. 2019).
-   `POMO`: Policy Optimization with Multiple Optima (Kwon et al. 2020) via `AttentionModel` configuration.
-   `HeterogeneousAttentionModel` (HAM): For problems with multiple node types (PDP).
-   `L2DModel`: Learning to Dispatch for Job Shop Scheduling (JSSP).
-   `PtrNet`: Pointer Network (Vinyals et al. 2015).

Iterative Improvement Models:
-   `NeuOpt`: Neural Optimizer for VRP.
-   `N2S`: Neural Neighborhood Search.
-   `DACT`: Dual-Aspect Collaborative Transformer.
-   `DeepACO`: Deep Ant Colony Optimization.

Transductive / Matrix Models:
-   `MatNet`: Matrix Encoding Networks (for ATsp/SDVRP).
-   `EAS`: Efficient Active Search (Hottung et al. 2022).

Baselines:
-   `CriticNetwork`: Value function approximation for REINFORCE.
"""

from logic.src.pipeline.rl.common.baselines import (
    Baseline as Baseline,
)
from logic.src.pipeline.rl.common.baselines import (
    CriticBaseline as CriticBaseline,
)
from logic.src.pipeline.rl.common.baselines import (
    ExponentialBaseline as ExponentialBaseline,
)
from logic.src.pipeline.rl.common.baselines import (
    NoBaseline as NoBaseline,
)
from logic.src.pipeline.rl.common.baselines import (
    POMOBaseline as POMOBaseline,
)
from logic.src.pipeline.rl.common.baselines import (
    RolloutBaseline as RolloutBaseline,
)
from logic.src.pipeline.rl.common.baselines import (
    WarmupBaseline as WarmupBaseline,
)

from .attention_model import AttentionModel as AttentionModel
from .critic_network import CriticNetwork as CriticNetwork
from .dact import DACT as DACT
from .deepaco import DeepACO as DeepACO
from .gat_lstm_manager import GATLSTManager as GATLSTManager
from .gfacs import GFACS as GFACS
from .glop import GLOP as GLOP
from .hypernet import (
    Hypernetwork as Hypernetwork,
)
from .hypernet import (
    HypernetworkOptimizer as HypernetworkOptimizer,
)
from .l2d import L2DModel as L2DModel
from .l2d import L2DPPOModel as L2DPPOModel
from .matnet import MatNet as MatNet
from .mdam import MDAM as MDAM
from .meta_rnn import WeightAdjustmentRNN as WeightAdjustmentRNN
from .n2s import N2S as N2S
from .nargnn import NARGNN as NARGNN
from .neuopt import NeuOpt as NeuOpt
from .policies.common.transductive import EAS as EAS
from .policies.common.transductive import ActiveSearch as ActiveSearch
from .policies.common.transductive import TransductiveModel as TransductiveModel
from .polynet import PolyNet as PolyNet
from .subnets import (
    GatedGraphAttConvEncoder as GatedGraphAttConvEncoder,
)
from .subnets import (
    GatedRecurrentFillPredictor as GatedRecurrentFillPredictor,
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
from .temporal_am import TemporalAttentionModel as TemporalAttentionModel

__all__ = [
    # Core Models
    "AttentionModel",
    "TemporalAttentionModel",
    "CriticNetwork",
    "GATLSTManager",
    "WeightAdjustmentRNN",
    "Hypernetwork",
    "HypernetworkOptimizer",
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
    "TransductiveModel",
    "ActiveSearch",
    "EAS",
    # Encoders
    "GraphAttentionEncoder",
    "GraphAttConvEncoder",
    "GatedGraphAttConvEncoder",
    "TransGraphConvEncoder",
    # Predictors
    "GatedRecurrentFillPredictor",
    # Baselines
    "Baseline",
    "NoBaseline",
    "ExponentialBaseline",
    "RolloutBaseline",
    "CriticBaseline",
    "POMOBaseline",
    "WarmupBaseline",
]
