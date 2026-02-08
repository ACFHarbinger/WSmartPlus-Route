"""
Policies module for WSmart-Route.

Contains all neural and classical policy implementations for combinatorial
optimization. Use ``get_policy(name)`` to look up a policy class by its
short CLI name, or import the class directly.

Registries:
- ``POLICY_REGISTRY``: Maps short names (e.g. "am", "mdam") to policy classes.
- ``get_policy(name, **kwargs)``: Factory function for instantiation.
"""

from typing import TYPE_CHECKING

import torch.nn as nn

from logic.src.models.attention_model.deep_decoder_policy import DeepDecoderPolicy
from logic.src.models.attention_model.policy import AttentionModelPolicy
from logic.src.models.attention_model.symnco_policy import SymNCOPolicy
from logic.src.models.common import (
    ConstructivePolicy,
    ImprovementPolicy,
    NonAutoregressivePolicy,
)
from logic.src.models.dact.policy import DACTPolicy
from logic.src.models.deepaco.policy import DeepACOPolicy
from logic.src.models.gfacs.policy import GFACSPolicy
from logic.src.models.glop.policy import GLOPPolicy
from logic.src.models.mdam.policy import MDAMPolicy
from logic.src.models.moe.policy import MoEPolicy
from logic.src.models.n2s.policy import N2SPolicy
from logic.src.models.nargnn.policy import NARGNNPolicy
from logic.src.models.neuopt.policy import NeuOptPolicy
from logic.src.models.pointer_network.policy import PointerNetworkPolicy
from logic.src.models.polynet.policy import PolyNetPolicy
from logic.src.models.temporal_attention_model.policy import TemporalAMPolicy

from .alns import VectorizedALNS
from .ant_colony_system import VectorizedACOPolicy
from .hgs import VectorizedHGS
from .hgs_alns import VectorizedHGSALNS
from .hybrid import NeuralHeuristicHybrid
from .iterated_local_search import IteratedLocalSearchPolicy

# Short-name registry: CLI model name -> (module_path, class_name)
# This enables ``get_policy("am")`` without importing everything eagerly.
_POLICY_REGISTRY_SPEC = {
    "am": ("logic.src.models.attention_model.policy", "AttentionModelPolicy"),
    "deep_decoder": ("logic.src.models.attention_model.deep_decoder_policy", "DeepDecoderPolicy"),
    "deepaco": ("logic.src.models.deepaco.policy", "DeepACOPolicy"),
    "gfacs": ("logic.src.models.gfacs.policy", "GFACSPolicy"),
    "glop": ("logic.src.models.glop.policy", "GLOPPolicy"),
    "mdam": ("logic.src.models.mdam.policy", "MDAMPolicy"),
    "moe": ("logic.src.models.moe.policy", "MoEPolicy"),
    "nargnn": ("logic.src.models.nargnn.policy", "NARGNNPolicy"),
    "n2s": ("logic.src.models.n2s.policy", "N2SPolicy"),
    "neuopt": ("logic.src.models.neuopt.policy", "NeuOptPolicy"),
    "dact": ("logic.src.models.dact.policy", "DACTPolicy"),
    "pointer": ("logic.src.models.pointer_network.policy", "PointerNetworkPolicy"),
    "polynet": ("logic.src.models.polynet.policy", "PolyNetPolicy"),
    "symnco": ("logic.src.models.attention_model.symnco_policy", "SymNCOPolicy"),
    "temporal": ("logic.src.models.temporal_attention_model.policy", "TemporalAMPolicy"),
}


def get_policy_class(name: str) -> type:
    """
    Look up a policy class by its short CLI name.

    Args:
        name: Short policy name (e.g. "am", "mdam", "deepaco").

    Returns:
        The policy class (not instantiated).

    Raises:
        ValueError: If the name is not found in the registry.
    """
    import importlib

    if name not in _POLICY_REGISTRY_SPEC:
        raise ValueError(f"Unknown policy: {name!r}. Available: {sorted(_POLICY_REGISTRY_SPEC.keys())}")
    module_path, class_name = _POLICY_REGISTRY_SPEC[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_policy(name: str, **kwargs) -> nn.Module:
    """
    Create a policy instance by short CLI name.

    Args:
        name: Short policy name (e.g. "am", "mdam", "deepaco").
        **kwargs: Arguments passed to the policy constructor.

    Returns:
        Instantiated policy module.
    """
    cls = get_policy_class(name)
    return cls(**kwargs)


__all__ = [
    "ConstructivePolicy",
    "ImprovementPolicy",
    "AttentionModelPolicy",
    "DeepDecoderPolicy",
    "DeepACOPolicy",
    "GFACSPolicy",
    "GLOPPolicy",
    "MDAMPolicy",
    "NARGNNPolicy",
    "PolyNetPolicy",
    "TemporalAMPolicy",
    "PointerNetworkPolicy",
    "SymNCOPolicy",
    "MoEPolicy",
    "NeuOptPolicy",
    "DACTPolicy",
    "N2SPolicy",
    "NonAutoregressivePolicy",
    "get_policy_class",
    "get_policy",
    "VectorizedHGSALNS",
    "IteratedLocalSearchPolicy",
    "VectorizedACOPolicy",
    "VectorizedALNS",
    "VectorizedHGS",
    "NeuralHeuristicHybrid",
]
