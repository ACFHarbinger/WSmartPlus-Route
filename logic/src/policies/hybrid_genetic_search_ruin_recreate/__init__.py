"""
Hybrid Genetic Search with Ruin-and-Recreate (HGSRR) policy module.

This module combines the evolutionary framework of HGS with the adaptive
destroy/repair operators from ALNS (Ruin-and-Recreate paradigm).

Reference:
    Pisinger, D., & Ropke, S. (2019). Large neighborhood search.
    In Handbook of metaheuristics (pp. 99-127). Springer, Cham.
"""

from .hgsrr import run_hgsrr
from .params import HGSRRParams

__all__ = ["run_hgsrr", "HGSRRParams"]
