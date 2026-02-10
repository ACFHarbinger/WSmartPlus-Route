"""
Base classes and utilities for simulation actions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from logic.src.interfaces import ITraversable


def _flatten_config(cfg: Any) -> dict:
    """
    Helper to flatten nested configuration structures (e.g. hgs.custom -> list of dicts).
    """
    if not cfg:
        return {}

    curr = cfg
    # Unwrap single-key nested dicts (Hydra structure often starts with policy name)
    while isinstance(curr, dict) and len(curr) == 1:
        key = next(iter(curr))
        # If we reached the target object itself, stop unwrapping
        if key in ["must_go", "policy", "post_processing"]:
            break
        curr = curr[key]

    # Handle list of dicts (common in Hydra 'custom' lists)
    if isinstance(curr, list):
        merged = {}
        for item in curr:
            if isinstance(item, ITraversable):
                merged.update(item)
        return merged

    # Handle dict which might contain lists to be flattened (e.g. {'custom': [...], 'ortools': [...]})
    if isinstance(curr, ITraversable):
        flat = {**curr}
        # Iterate over all keys and flatten if value is a list of dicts
        for _k, v in list(flat.items()):
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, ITraversable):
                        flat.update(item)
        return flat

    return {}


class SimulationAction(ABC):
    """
    Abstract base class for simulation day actions.

    Defines the interface for all simulation commands. Each action receives
    a shared context dictionary and modifies it in-place with its outputs.
    """

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> None:
        """
        Executes the action and updates the context in-place.

        Args:
            context: Shared dictionary containing simulation state.
        """
        pass
