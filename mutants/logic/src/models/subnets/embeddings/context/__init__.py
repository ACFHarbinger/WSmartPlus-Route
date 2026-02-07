"""
Unified context embedding modules for VRP variants.
"""

from .context_base import EnvContext
from .cvrpp import CVRPPContext
from .embedder_base import ContextEmbedder
from .generic import GenericContextEmbedder
from .swcvrp_context import SWCVRPContext
from .vrpp_context import VRPPContext
from .vrpp_embedder import VRPPContextEmbedder
from .wc_embedder import WCContextEmbedder
from .wcvrp_context import WCVRPContext

__all__ = [
    "ContextEmbedder",
    "EnvContext",
    "GenericContextEmbedder",
    "VRPPContextEmbedder",
    "VRPPContext",
    "CVRPPContext",
    "WCContextEmbedder",
    "WCVRPContext",
    "SWCVRPContext",
]
