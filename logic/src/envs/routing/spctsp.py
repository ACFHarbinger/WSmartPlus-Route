"""
SPCTSP Environment — Stochastic Prize-Collecting TSP.

Identical to PCTSPEnv except that the prize revealed upon visiting a node
is drawn from a stochastic distribution rather than being deterministic.
The expected prize is still made available to the policy for conditioning.

Attributes:
    NAME: Name of the environment.
    name: Name of the environment.
    node_dim: Node feature dimension (x, y coordinates only → 2).
    _stochastic: Boolean flag indicating that this is a stochastic environment.

Example:
    >>> from logic.src.envs.routing import get_env
    >>> env = get_env("spctsp", num_loc=50)
    >>> td = env.reset()
    >>> td = env.step(td)
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from logic.src.envs.generators.pctsp import PCTSPGenerator
from logic.src.envs.routing.pctsp import PCTSPEnv


class SPCTSPEnv(PCTSPEnv):
    """
    Stochastic Prize-Collecting Traveling Salesman Problem (SPCTSP).

    The only difference from deterministic PCTSP is that the prize
    actually received when visiting a node (``real_prize``) is drawn
    from a uniform distribution centred on the expected prize, rather
    than being equal to it.  The expected prize is still observed.

    Attributes:
        NAME: Environment identifier string ``"spctsp"``.
        name: Alias for NAME used by the registry.
        node_dim: Node feature dimension — 2 for (x, y) coordinates.
    """

    NAME: str = "spctsp"
    name: str = "spctsp"
    _stochastic: bool = True

    def __init__(
        self,
        generator: Optional[PCTSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the SPCTSP environment.

        Args:
            generator: Pre-built PCTSPGenerator; created from generator_params if None.
            generator_params: Keyword arguments forwarded to PCTSPGenerator constructor.
            device: Torch device for tensor placement.
            kwargs: Additional arguments forwarded to RL4COEnvBase.
        """
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            device=device,
            **kwargs,
        )
