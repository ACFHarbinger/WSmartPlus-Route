"""
Global Cut Pool for Branch-and-Price-and-Cut.
"""

from dataclasses import dataclass
from typing import Any, List


@dataclass
class CutInfo:
    """Metadata describing a generated valid inequality."""

    type: str  # e.g., 'rcc', 'sec', 'sri', 'lci'
    data: Any  # node-set, RHS, lifting coeffs, etc.
    active: bool = True
    violation: float = 0.0


class GlobalCutPool:
    """
    Centralized repository for globally valid inequalities across B&B nodes.

    Philosophy:
    In BPC, separation is expensive. By pooling valid inequalities (RCC, SRI, SEC 2.1)
    globally, we ensure that a cut discovered in one branch tightens the LP bound
    in sibling and child branches immediately, avoiding redundant separation and
    reducing the total number of B&B nodes explored.

    Global Validity Contract:
        Only archive cuts here that are valid at EVERY node in the B&B tree
        (e.g., RCC, SRI, SEC 2.1). Node-local cuts (branching-dependent SECs,
        lifted cover cuts dependent on local bounds) MUST NOT be added to
        this pool or they will lead to incorrect pruning and loss of optimality.

    Assumption:
        Customer demands and vehicle capacities are static throughout the B&B tree.
        If these were dynamic, the RHS of node-set-based cuts could change,
        invalidating the global mathematical integrity of this archive.
    """

    def __init__(self) -> None:
        self.cuts: List[CutInfo] = []
        self._keys: set = set()

    def add_cut(self, cut_type: str, data: Any) -> bool:
        """
        Archive a cut if it's not already in the pool.
        """
        # For simple cuts like SEC/RCC, data is the FrozenSet of nodes
        # Using a unified key check to prevent redundant storage
        key = (cut_type, str(data))
        if key in self._keys:
            return False

        self.cuts.append(CutInfo(type=cut_type, data=data))
        self._keys.add(key)
        return True

    def get_cuts_by_type(self, cut_type: str) -> List[CutInfo]:
        """Retrieve all archived cuts of a specific category."""
        return [c for c in self.cuts if c.type == cut_type]

    def clear(self) -> None:
        """Purge the entire pool."""
        self.cuts.clear()
        self._keys.clear()
