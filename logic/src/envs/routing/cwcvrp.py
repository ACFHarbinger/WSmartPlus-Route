"""
CWCVRP Environment implementation.

Attributes:
    CWCVRPEnv: CWCVRP environment.

Example:
    >>> from logic.src.envs.routing import get_env
    >>> env = get_env("cwcvrp", num_loc=50)
    >>> td = env.reset()
"""

from __future__ import annotations

from logic.src.envs.routing.wcvrp import WCVRPEnv


class CWCVRPEnv(WCVRPEnv):
    """Capacitated WCVRP with strict capacity constraints.

    Attributes:
        name: Name of the environment.
    """

    name: str = "cwcvrp"
