"""
CWCVRP Environment implementation.
"""

from __future__ import annotations

from logic.src.envs.routing.wcvrp import WCVRPEnv


class CWCVRPEnv(WCVRPEnv):
    """Capacitated WCVRP with strict capacity constraints."""

    name: str = "cwcvrp"
