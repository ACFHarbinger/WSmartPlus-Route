"""
SPCTSP Environment — Stochastic Prize-Collecting TSP.

Identical to PCTSPEnv except that the prize revealed upon visiting a node
is drawn from a stochastic distribution rather than being deterministic.
The expected prize is still made available to the policy for conditioning.
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
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            device=device,
            **kwargs,
        )
