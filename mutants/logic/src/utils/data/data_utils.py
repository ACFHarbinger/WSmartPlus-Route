"""
Backward compatibility shim for data_utils.
"""

from .generation import generate_waste_prize
from .loader import (
    check_extension,
    collate_fn,
    load_dataset,
    load_focus_coords,
    load_td_dataset,
    save_dataset,
    save_td_dataset,
)
from .parameters import load_area_and_waste_type_params

__all__ = [
    "check_extension",
    "save_dataset",
    "save_td_dataset",
    "load_td_dataset",
    "load_dataset",
    "collate_fn",
    "load_focus_coords",
    "generate_waste_prize",
    "load_area_and_waste_type_params",
]
