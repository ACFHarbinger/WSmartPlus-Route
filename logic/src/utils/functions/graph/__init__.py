"""
Facade for graph utilities.
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
]
