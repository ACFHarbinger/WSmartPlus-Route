"""Hybrid Attention Model architectures.

This package provides models that combine attention mechanisms with classical
heuristics, including two-stage policies and neural operator selection.

Attributes:
    HybridTwoStagePolicy: Unified init-refinement model.
    NeuralHeuristicHybrid: Simple constructive-refinement wrapper.
    ImprovementStepDecoder: Operator selection network.
"""

from .hybrid_neural_heuristic_policy import NeuralHeuristicHybrid as NeuralHeuristicHybrid
from .hybrid_two_step_policy import HybridTwoStagePolicy as HybridTwoStagePolicy
from .improvement_step_decoder import ImprovementStepDecoder as ImprovementStepDecoder

__all__ = [
    "HybridTwoStagePolicy",
    "NeuralHeuristicHybrid",
    "ImprovementStepDecoder",
]
