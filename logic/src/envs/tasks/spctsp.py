"""
SPCTSP problem definition for offline evaluation.
"""

from __future__ import annotations

from logic.src.envs.tasks.pctsp import PCTSP


class SPCTSP(PCTSP):
    """
    Stochastic Prize-Collecting TSP (SPCTSP).

    Identical cost computation to PCTSP; uses ``stochastic_prize`` rather
    than ``deterministic_prize`` when evaluating collected prize.
    """

    NAME = "spctsp"
