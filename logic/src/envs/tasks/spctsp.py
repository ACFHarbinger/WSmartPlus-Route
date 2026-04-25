"""
SPCTSP problem definition for offline evaluation.

Attributes:
    SPCTSP: Stochastic Prize-Collecting TSP (SPCTSP) definition.

Example:
    >>> import torch
    >>> from logic.src.envs.tasks.spctsp import SPCTSP
    >>> dataset = {
    ...     "locs": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
    ...     "penalty": torch.tensor([[0.0, 10.0]]),
    ...     "real_prize": torch.tensor([[0.0, 10.0]]),
    ...     "depot": torch.tensor([0.0]),
    ... }
    >>> pi = torch.tensor([[[0, 1, 0]]])
    >>> length, cost_dict, _ = SPCTSP.get_costs(dataset, pi)
    >>> print(length)
    tensor([-2.0])
"""

from __future__ import annotations

from logic.src.envs.tasks.pctsp import PCTSP


class SPCTSP(PCTSP):
    """
    Stochastic Prize-Collecting TSP (SPCTSP).

    Identical cost computation to PCTSP; uses ``stochastic_prize`` rather
    than ``deterministic_prize`` when evaluating collected prize.

    Attributes:
        NAME: Environment name identifier.
    """

    NAME = "spctsp"
