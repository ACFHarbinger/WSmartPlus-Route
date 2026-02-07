"""
Policies module for WSmart-Route.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from logic.src.models.policies.common import (
    ConstructivePolicy,
    ImprovementPolicy,
)

if TYPE_CHECKING:
    from .am import AttentionModelPolicy
    from .common.nonautoregressive import NonAutoregressivePolicy
    from .dact import DACTPolicy
    from .deep_decoder import DeepDecoderPolicy
    from .deepaco import DeepACOPolicy
    from .gfacs import GFACSPolicy
    from .glop import GLOPPolicy
    from .mdam import MDAMPolicy
    from .moe import MoEPolicy
    from .n2s import N2SPolicy
    from .nargnn import NARGNNPolicy
    from .neuopt import NeuOptPolicy
    from .pointer import PointerNetworkPolicy
    from .polynet import PolyNetPolicy
    from .symnco import SymNCOPolicy
    from .temporal import TemporalAMPolicy

_POLICY_MAP = {
    "AttentionModelPolicy": ".am",
    "DeepDecoderPolicy": ".deep_decoder",
    "DeepACOPolicy": ".deepaco",
    "GFACSPolicy": ".gfacs",
    "GLOPPolicy": ".glop",
    "MDAMPolicy": ".mdam",
    "MoEPolicy": ".moe",
    "NARGNNPolicy": ".nargnn",
    "N2SPolicy": ".n2s",
    "NeuOptPolicy": ".neuopt",
    "DACTPolicy": ".dact",
    "NonAutoregressivePolicy": ".common.nonautoregressive",
    "PointerNetworkPolicy": ".pointer",
    "PolyNetPolicy": ".polynet",
    "SymNCOPolicy": ".symnco",
    "TemporalAMPolicy": ".temporal",
}


def __getattr__(name):
    if name in _POLICY_MAP:
        import importlib

        module_path = _POLICY_MAP[name]
        module = importlib.import_module(module_path, __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
]
