"""WCVRP specific context embedding module.

This module provides the WCVRPState component, which builds on VRPPState
to provide context for Waste Collection VRP variants.

Attributes:
    WCVRPState: State encoder for Waste Collection VRPs.

Example:
    >>> from logic.src.models.subnets.embeddings.state.wcvrp import WCVRPState
    >>> state_embedder = WCVRPState(embed_dim=128)
    >>> context = state_embedder(embeddings, td)
"""

from __future__ import annotations

from .vrpp import VRPPState


class WCVRPState(VRPPState):
    """Context embedding for WCVRP.

    Inherits from VRPPState as the fundamental state features (capacity/remaining
    length) are shared between these problem types.

    Attributes:
        embed_dim (int): Dimensionality of the projected state context.
    """

    pass
