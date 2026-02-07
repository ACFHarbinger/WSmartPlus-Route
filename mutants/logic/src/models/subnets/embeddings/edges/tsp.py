from .base import EdgeEmbedding


class TSPEdgeEmbedding(EdgeEmbedding):
    """
    Edge embedding for TSP-like problems.

    Uses pairwise Euclidean distances as edge features, optionally
    sparsified to k-nearest neighbors per node.
    """

    node_dim = 1
