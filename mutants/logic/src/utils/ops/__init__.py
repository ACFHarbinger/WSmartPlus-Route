"""
Unified tensor operations for combinatorial optimization.

This package provides commonly-used tensor operations for VRP/CO problems.
"""

from .distance import (
    get_distance,
    get_distance_matrix,
    get_open_tour_length,
    get_tour_length,
)
from .graph import (
    get_full_graph_edge_index,
    sparsify_graph,
)
from .pomo import (
    get_best_actions,
    get_num_starts,
    select_start_nodes,
    select_start_nodes_by_distance,
)
from .probabilistic import calculate_entropy
from .tensor import unbatchify_and_gather

__all__ = [
    "get_distance",
    "get_distance_matrix",
    "get_tour_length",
    "get_open_tour_length",
    "calculate_entropy",
    "select_start_nodes",
    "select_start_nodes_by_distance",
    "get_num_starts",
    "get_best_actions",
    "unbatchify_and_gather",
    "sparsify_graph",
    "get_full_graph_edge_index",
]
