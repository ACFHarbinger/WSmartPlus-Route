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
from logic.src.models.policies.common import (
    ConstructivePolicy,
    ImprovementPolicy,
)

if TYPE_CHECKING:
    from .am import AttentionModelPolicy
    from .common import NonAutoregressivePolicy
    from .dact_policy import DACTPolicy
    from .deep_decoder import DeepDecoderPolicy
    from .deepaco import DeepACOPolicy
    from .gfacs import GFACSPolicy
    from .glop import GLOPPolicy
    from .mdam import MDAMPolicy
    from .moe import MoEPolicy
    from .n2s_policy import N2SPolicy
    from .nargnn import NARGNNPolicy
    from .neuopt_policy import NeuOptPolicy
    from .pointer import PointerNetworkPolicy
    from .polynet import PolyNetPolicy
    from .symnco import SymNCOPolicy
    from .temporal import TemporalAMPolicy


# Lazy-loading map: class name -> relative module path
_POLICY_MAP = {
    "AttentionModelPolicy": ".am",
    "DeepDecoderPolicy": ".deep_decoder",
    "DeepACOPolicy": ".deepaco",
    "GFACSPolicy": ".gfacs",
    "GLOPPolicy": ".glop",
    "MDAMPolicy": ".mdam",
    "MoEPolicy": ".moe",
    "NARGNNPolicy": ".nargnn",
    "N2SPolicy": ".n2s_policy",
    "NeuOptPolicy": ".neuopt_policy",
    "DACTPolicy": ".dact_policy",
    "NonAutoregressivePolicy": ".common",
    "PointerNetworkPolicy": ".pointer",
    "PolyNetPolicy": ".polynet",
    "SymNCOPolicy": ".symnco",
    "TemporalAMPolicy": ".temporal",
}

# Short-name registry: CLI model name -> (module_path, class_name)
# This enables ``get_policy("am")`` without importing everything eagerly.
_POLICY_REGISTRY_SPEC = {
    "am": (".am", "AttentionModelPolicy"),
    "deep_decoder": (".deep_decoder", "DeepDecoderPolicy"),
    "deepaco": (".deepaco", "DeepACOPolicy"),
    "gfacs": (".gfacs", "GFACSPolicy"),
    "glop": (".glop", "GLOPPolicy"),
    "ham": (".ham", "HAMPolicy"),
    "l2d": (".l2d_policy", "L2DPolicy"),
    "mdam": (".mdam", "MDAMPolicy"),
    "moe": (".moe", "MoEPolicy"),
    "nargnn": (".nargnn", "NARGNNPolicy"),
    "n2s": (".n2s_policy", "N2SPolicy"),
    "neuopt": (".neuopt_policy", "NeuOptPolicy"),
    "dact": (".dact_policy", "DACTPolicy"),
    "pointer": (".pointer", "PointerNetworkPolicy"),
    "polynet": (".polynet", "PolyNetPolicy"),
    "symnco": (".symnco", "SymNCOPolicy"),
    "temporal": (".temporal", "TemporalAMPolicy"),
}


def __getattr__(name):
    if name in _POLICY_MAP:
        import importlib

        module_path = _POLICY_MAP[name]
        module = importlib.import_module(module_path, __package__)
        return getattr(module, name)

    if name == "POLICY_REGISTRY":
        # Lazily populate the full registry on first access
        return {k: get_policy_class(k) for k in _POLICY_REGISTRY_SPEC}

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    module = importlib.import_module(module_path, __package__)
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
]
