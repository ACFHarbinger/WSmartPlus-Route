"""
Exchange operators for VRP local search.

This module contains advanced inter-route and intra-route operators:
- Or-opt: Relocate chains of consecutive nodes
- Cross-Exchange: Swap segments between routes (λ-interchange)
- Ejection Chain: Compound displacement for fleet reduction

(Refactored to point to `logic.src.policies.operators.exchange` package)
"""

from .exchange import (
    move_or_opt,
)

__all__ = [
    "move_or_opt",
]
