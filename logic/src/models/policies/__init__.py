"""
Policies module for WSmart-Route.
"""

from logic.src.models.policies.common import (
    ConstructivePolicy,
    ImprovementPolicy,
)

_POLICY_MAP = {
    "AttentionModelPolicy": ".am",
    "DeepDecoderPolicy": ".deep_decoder",
    "DeepACOPolicy": ".deepaco",
    "GFACSPolicy": ".gfacs",
    "GLOPPolicy": ".glop",
    "MDAMPolicy": ".mdam",
    "MoEPolicy": ".moe",
    "NARGNNPolicy": ".nargnn",
    "NeuOptPolicy": ".neuopt",
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
]
