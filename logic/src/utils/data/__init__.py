"""
Data utilities module.

Attributes:
    check_extension: Check if the file extension is supported.
    collate_fn: Collate list of TensorDicts or dicts into batched TensorDict or dict.
    load_dataset: Load a dataset from a file.
    load_focus_coords: Load focus coordinates from a file.
    load_grid_base: Load grid base from a file.
    load_simulation_dataset: Load a simulation dataset from a file.
    load_td_dataset: Load a TensorDict dataset from a file.
    save_dataset: Save a dataset to a file.
    save_simulation_dataset: Save a simulation dataset to a file.
    save_td_dataset: Save a TensorDict dataset to a file.
    TensorDictStateWrapper: A wrapper for TensorDict that exposes methods expected by legacy model components (Decoders).

Example:
    >>> from logic.src.utils.data import (
    ...     check_extension,
    ...     collate_fn,
    ...     load_dataset,
    ...     load_focus_coords,
    ...     load_grid_base,
    ...     load_simulation_dataset,
    ...     load_td_dataset,
    ...     save_dataset,
    ...     save_simulation_dataset,
    ...     save_td_dataset,
    ...     TensorDictStateWrapper,
    ... )
"""

from .loader import (
    check_extension,
    collate_fn,
    load_dataset,
    load_focus_coords,
    load_grid_base,
    load_simulation_dataset,
    load_td_dataset,
    save_dataset,
    save_simulation_dataset,
    save_td_dataset,
)
from .td_state_wrapper import TensorDictStateWrapper

__all__ = [
    "check_extension",
    "save_dataset",
    "save_simulation_dataset",
    "save_td_dataset",
    "load_td_dataset",
    "load_dataset",
    "load_simulation_dataset",
    "collate_fn",
    "load_focus_coords",
    "load_grid_base",
    "TensorDictStateWrapper",
]
