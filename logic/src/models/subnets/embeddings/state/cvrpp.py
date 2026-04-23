"""CVRPP specific context embedding module.

This module provides the CVRPPState component, which builds on VRPPState
to provide context for Capacitated VRPs with Profits.

Attributes:
    CVRPPState: State encoder for Capacitated VRPPs.

Example:
    >>> from logic.src.models.subnets.embeddings.state.cvrpp import CVRPPState
    >>> state_embedder = CVRPPState(embed_dim=128)
    >>> context = state_embedder(embeddings, td)
"""

from __future__ import annotations

from .vrpp import VRPPState


class CVRPPState(VRPPState):
    """Context embedding for CVRPP.

    Inherits from VRPPState as the fundamental state features (capacity/remaining
    length) are shared between these problem types.

    Attributes:
        embed_dim (int): Dimensionality of the projected state context.
    """

    pass
