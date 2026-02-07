"""
VRPy engine for Branch-Cut-and-Price module.
"""

import logging

import networkx as nx
from vrpy import VehicleRoutingProblem


def run_bcp_vrpy(dist_matrix, demands, capacity, R, C, values):
    """
    Solve CVRP using VRPy (Column Generation / Branch-and-Price).

    Note: VRPy does not natively support Prize-Collecting CVRP via simple
    configuration. This implementation solves standard CVRP for ALL nodes
    present in the demands dictionary (no node dropping).

    Uses NetworkX DiGraph representation with:
    - Source/Sink virtual nodes for depot
    - Edge costs scaled by C coefficient
    - Load capacity constraints
    - Column generation solver from VRPy library

    Args:
        dist_matrix (np.ndarray): Distance matrix (N x N)
        demands (dict): Node demands {node_id: demand_value}
        capacity (float): Vehicle capacity
        R (float): Revenue per unit demand (unused in this variant)
        C (float): Cost per unit distance
        values (dict): Config with 'time_limit' (default: 30)

    Returns:
        Tuple[List[List[int]], float]: Routes and total cost
            - routes: List of routes excluding Source/Sink
            - cost: Total objective value from VRPy
    """
    if VehicleRoutingProblem is None:
        logging.error("VRPy not installed or import failed.")
        return [], 0.0

    # Suppress Logs
    logging.getLogger("cspy").setLevel(logging.WARNING)
    logging.getLogger("vrpy").setLevel(logging.WARNING)

    G = nx.DiGraph()
    n_nodes = len(dist_matrix) - 1

    # Add Nodes
    for i in range(1, n_nodes + 1):
        d = demands.get(i, 0)
        G.add_node(i, demand=d)

    # Add Edges
    for i in range(1, n_nodes + 1):
        cost = dist_matrix[0][i] * C
        G.add_edge("Source", i, cost=cost)

    for i in range(1, n_nodes + 1):
        cost = dist_matrix[i][0] * C
        G.add_edge(i, "Sink", cost=cost)

    for i in range(1, n_nodes + 1):
        for j in range(1, n_nodes + 1):
            if i != j:
                cost = dist_matrix[i][j] * C
                G.add_edge(i, j, cost=cost)

    prob = VehicleRoutingProblem(G, load_capacity=capacity)

    time_limit = values.get("time_limit", 30)
    prob.solve(time_limit=time_limit)

    if prob.best_routes:
        routes = []
        for r_id, path in prob.best_routes.items():
            clean_route = [node for node in path if node != "Source" and node != "Sink"]
            if clean_route:
                routes.append(clean_route)
        return routes, prob.best_value
    else:
        return [], 0.0
