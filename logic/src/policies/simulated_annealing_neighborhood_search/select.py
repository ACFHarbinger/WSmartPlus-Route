"""
Facade for the select package.
"""

from logic.src.policies.look_ahead_aux.select import (
    add_bin,
    add_n_bins_consecutive,
    add_n_bins_random,
    add_route_consecutive,
    add_route_random,
    add_route_with_removed_bins_consecutive,
    add_route_with_removed_bins_random,
    insert_bins,
    remove_bin,
    remove_bins_end,
    remove_n_bins_consecutive,
    remove_n_bins_random,
)

__all__ = [
    "remove_bin",
    "add_bin",
    "insert_bins",
    "remove_bins_end",
    "remove_n_bins_random",
    "remove_n_bins_consecutive",
    "add_n_bins_random",
    "add_n_bins_consecutive",
    "add_route_random",
    "add_route_consecutive",
    "add_route_with_removed_bins_random",
    "add_route_with_removed_bins_consecutive",
]
