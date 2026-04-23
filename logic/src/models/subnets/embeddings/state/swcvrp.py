"""SWCVRP specific context embedding module.

This module provides the SWCVRPState component, which extends WCVRPState
to handle context for Stochastic Waste Collection VRPs.

Attributes:
    SWCVRPState: State encoder for Stochastic Waste Collection VRPs.

Example:
    >>> from logic.src.models.subnets.embeddings.state.swcvrp import SWCVRPState
    >>> state_embedder = SWCVRPState(embed_dim=128)
    >>> context = state_embedder(embeddings, td)
"""

from __future__ import annotations

from .wcvrp import WCVRPState


class SWCVRPState(WCVRPState):
    """Context embedding for SWCVRP (Stochastic WCVRP).

    Inherits from WCVRPState as the fundamental state features (capacity/remaining
    length) are shared between these problem types.

    Attributes:
        embed_dim (int): Dimensionality of the projected state context.
    """

    pass
