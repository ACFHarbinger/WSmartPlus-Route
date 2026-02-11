"""
Base classes and utilities for simulation actions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


def _find_must_go(d: Any) -> Any:
    """
    Recursively find the first occurrence of 'must_go' in a nested structure.
    Handles dictionaries, ITraversable objects, and iterable sequences (lists, ListConfig, etc.).
    """
    if isinstance(d, dict):
        if "must_go" in d:
            return d["must_go"]
        for val in d.values():
            res = _find_must_go(val)
            if res is not None:
                return res
    elif hasattr(d, "items"):  # Handle ITraversable/Hydra objects
        d_dict = dict(d)
        if "must_go" in d_dict:
            return d_dict["must_go"]
        for val in d_dict.values():
            res = _find_must_go(val)
            if res is not None:
                return res
    elif isinstance(d, (list, tuple)) or (
        not isinstance(d, str) and hasattr(d, "__iter__")
    ):  # Handle sequences including ListConfig
        for item in d:
            res = _find_must_go(item)
            if res is not None:
                return res
    return None


def _flatten_config(cfg: Any) -> dict:
    """
    Helper to flatten nested configuration structures (e.g. hgs.custom -> list of dicts).
    """
    if not cfg:
        return {}

    curr = cfg
    # Unwrap single-key nested dicts (Hydra structure often starts with policy name)
    while (isinstance(curr, dict) or hasattr(curr, "items")) and len(curr) == 1:
        key = next(iter(curr.keys() if hasattr(curr, "keys") else curr))
        # If we reached the target object itself, stop unwrapping
        if key in ["must_go", "policy", "post_processing"]:
            break
        curr = curr[key]

    # Handle list of dicts (common in Hydra 'custom' lists)
    if isinstance(curr, list):
        merged: Dict[str, Any] = {}
        for item in curr:
            if hasattr(item, "items"):
                merged.update(dict(item))
        return merged

    # Handle dict which might contain lists to be flattened
    if hasattr(curr, "items"):
        flat = dict(curr)
        # Iterate over all keys and flatten if value is a list of dicts
        for _k, v in list(flat.items()):
            if isinstance(v, list):
                for item in v:
                    if hasattr(item, "items"):
                        flat.update(dict(item))

        if "must_go" not in flat:
            mg = _find_must_go(curr)
            if mg:
                flat["must_go"] = mg

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
