"""
Facade for graph utilities.

Attributes:
    find_longest_path: Find the longest path in a DAG represented by a distance matrix.
    adj_to_idx: Converts dimensionality-2 adjacency matrix to a [2, num_edges] index array.
    idx_to_adj: Converts edge index array back to an adjacency matrix.
    tour_to_adj: Converts a sequence of nodes (tour) into an adjacency matrix.
    sort_by_pairs: Sorts edge indices by their linear index (row * size + col).
    generate_adj_matrix: Generates a random adjacency matrix.
    get_edge_idx_dist: Generates edge indices based on shortest distances in the distance matrix.
    get_adj_knn: Generates an adjacency matrix based on K-Nearest Neighbors.
    get_adj_osm: Computes an adjacency matrix via OpenStreetMap for given coordinates.
    apply_edges: Sparsifies distance matrix and computes shortest paths.
    get_paths_between_states: Constructs a nested list of paths between all pairs of bins.

Example:
    find_longest_path(dist_matrix)
    adj_to_idx(adj_matrix)
    idx_to_adj(edge_idx)
    tour_to_adj(tour_nodes)
    sort_by_pairs(graph_size, edge_idx)
    generate_adj_matrix(size, num_edges)
    get_edge_idx_dist(dist_matrix, num_edges)
    get_adj_knn(dist_matrix, k_neighbors)
    get_adj_osm(coords, size, args)
    apply_edges(dist_matrix, edge_thresh, edge_method)
    get_paths_between_states(n_bins, shortest_paths)
"""

from .algorithms import find_longest_path as find_longest_path
from .conversion import (
    adj_to_idx as adj_to_idx,
)
from .conversion import (
    idx_to_adj as idx_to_adj,
)
from .conversion import (
    sort_by_pairs as sort_by_pairs,
)
from .conversion import (
    tour_to_adj as tour_to_adj,
)
from .generation import (
    generate_adj_matrix as generate_adj_matrix,
)
from .generation import (
    get_adj_knn as get_adj_knn,
)
from .generation import (
    get_adj_osm as get_adj_osm,
)
from .generation import (
    get_edge_idx_dist as get_edge_idx_dist,
)
from .network_utils import (
    apply_edges as apply_edges,
)
from .network_utils import (
    get_paths_between_states as get_paths_between_states,
)

__all__ = [
    "find_longest_path",
    "adj_to_idx",
    "idx_to_adj",
    "tour_to_adj",
    "sort_by_pairs",
    "generate_adj_matrix",
    "get_edge_idx_dist",
    "get_adj_knn",
    "get_adj_osm",
    "get_paths_between_states",
    "apply_edges",
]
