from .cvrpp import CVRPPEdgeEmbedding


class WCVRPEdgeEmbedding(CVRPPEdgeEmbedding):
    """
    Edge embedding for Waste Collection VRP problems.

    Inherits from CVRPEdgeEmbedding since WCVRP has the same graph
    structure (depot + customer nodes with capacity constraints).
    """

    pass
