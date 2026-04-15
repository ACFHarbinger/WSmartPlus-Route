"""
Base classes and utilities for simulation actions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping


def _find_mandatory(d: Any) -> Any:
    """
    Recursively find the first occurrence of 'mandatory' in a nested structure.
    Handles dictionaries, ITraversable objects, and iterable sequences (lists, ListConfig, etc.).
    """
    if isinstance(d, dict) or hasattr(d, "items"):
        d_dict = dict(d) if hasattr(d, "items") else d
        if "mandatory" in d_dict:
            return d_dict["mandatory"]
        for val in d_dict.values():
            res = _find_mandatory(val)
            if res is not None:
                return res
    elif isinstance(d, (list, tuple)) or (
        not isinstance(d, (str, dict)) and hasattr(d, "__iter__")
    ):  # Handle sequences including ListConfig
        for item in d:
            res = _find_mandatory(item)
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

    # Use a type guard to help Pyright
    while isinstance(curr, Mapping) and len(curr) == 1:
        # Safer key extraction
        key = next(iter(curr.keys()))

        if key in ["mandatory", "policy", "route_improvement"]:
            break
        curr = curr[key]

    # Handle list of dicts (including OmegaConf ListConfig)
    if isinstance(curr, (list, tuple)) or (not isinstance(curr, (str, dict, Mapping)) and hasattr(curr, "__iter__")):
        merged: Dict[str, Any] = {}
        for item in curr:
            if isinstance(item, Mapping):
                merged.update(item)  # No need to cast to dict() if it's already a Mapping
        return merged

    # Handle dict/Mapping
    if isinstance(curr, Mapping):
        flat = dict(curr)
        # Iterate over all keys and flatten if value is a list of dicts
        for _k, v in list(flat.items()):
            if isinstance(v, (list, tuple)) or (not isinstance(v, (str, dict)) and hasattr(v, "__iter__")):
                primitive_list = []
                for item in v:
                    if hasattr(item, "items"):
                        flat.update(dict(item))
                    else:
                        primitive_list.append(item)
                if primitive_list:
                    flat[_k] = primitive_list

        if "mandatory" not in flat:
            mg = _find_mandatory(curr)
            if mg:
                flat["mandatory"] = mg

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
