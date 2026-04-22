"""
Unified tensor operations for combinatorial optimization.

This package provides commonly-used tensor operations for VRP/CO problems.

Attributes:
    get_distance: Compute Euclidean distance between two batched coordinate tensors.
    get_distance_matrix: Compute pairwise Euclidean distance matrix from coordinates.
    get_tour_length: Compute total tour length for a batch of ordered location sequences.
    get_open_tour_length: Compute total tour length for open tours (no return to start).
    get_full_graph_edge_index: Get full graph edge index for a batch of node features.
    sparsify_graph: Sparsify graph by keeping only k-nearest neighbors for each node.
    select_start_nodes: Select start nodes for a batch of problems.
    select_start_nodes_by_distance: Select start nodes based on distance.
    get_num_starts: Get number of starts for a batch of problems.
    get_best_actions: Get best actions for a batch of problems.
    unbatchify_and_gather: Unbatchify and gather for a batch of problems.
    calculate_entropy: Calculate entropy for a batch of problems.

Example:
    >>> from logic.src.utils.ops import get_distance
    >>> x = torch.tensor([0.0, 0.0])
    >>> y = torch.tensor([3.0, 4.0])
    >>> get_distance(x, y)
    tensor(5.)
    >>> from logic.src.utils.ops.distance import get_distance_matrix
    >>> locs = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    >>> get_distance_matrix(locs)
    tensor([[[0., 5.], [5., 0.]]])
    >>> from logic.src.utils.ops.distance import get_tour_length
    >>> ordered_locs = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [6.0, 0.0]]])
    >>> get_tour_length(ordered_locs)
    tensor([16.])
    >>> from logic.src.utils.ops.distance import get_open_tour_length
    >>> get_open_tour_length(ordered_locs)
    tensor([10.])
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
