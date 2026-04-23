"""Core Neural Architectures for combinatorial optimization.

This package contains the primary model architectures used for solving
Vehicle Routing Problems, ranging from canonical attention-based seekers to
advanced meta-learning and iterative refinement models.

Attributes:
    AttentionModel: The standard graph attention-based policy.
    MatNet: The matrix-based encoder-decoder model.
    TemporalAttentionModel: AM with recurrent temporal forecasting.
    DACT: Collaborative transformer for iterative improvement.
    MDAM: Multi-decoder attention model for multi-objective search.
    MoE: Mixture-of-experts adaptive routing model.
    N2S: Neural neighborhood search for solution refinement.

Example:
    >>> from logic.src.models.core import AttentionModel
"""

from .attention_model import AttentionModel as AttentionModel
from .matnet import MatNet as MatNet

__all__ = [
    "AttentionModel",
    "MatNet",
]
