"""
Greedy Assignment Module for Fisher & Jaikumar Clustering.

This module implements the original greedy heuristic from Sultana & Akhand (2017)
for assigning nodes to seed clusters in the Cluster-First Route-Second algorithm.

The greedy method sorts all (node, seed) pairs by insertion cost and assigns
nodes to the first feasible seed that respects capacity constraints.

Reference:
    Sultana, T., Akhand, M. A. H., & Rahman, M. M. H. (2017). "A Variant Fisher
    and Jaikumar Algorithm to Solve Capacitated Vehicle Routing Problem".
"""

from typing import Dict, List

import numpy as np


def assign_greedy(
    seeds: List[int],
    must_go: List[int],
    wastes: Dict[int, float],
    capacity: float,
    distance_matrix: np.ndarray,
    strict_fleet: bool = False,
) -> List[List[int]]:
    """
    Assign nodes to seeds using the greedy heuristic from Sultana & Akhand (2017).

    This implements the original Fisher & Jaikumar greedy assignment strategy:
    1. Calculate insertion cost C_ik = d(i, 0) + d(k, i) - d(k, 0) for all (node, seed) pairs.
    2. Sort all pairs in increasing order of insertion cost.
    3. Assign each node to the first feasible seed (respecting capacity constraints).
    4. Handle unassigned nodes (behavior depends on strict_fleet parameter).

    Fleet Sizing Modes:
    - strict_fleet=False (Default): Dynamically open new vehicles if needed.
      This is suitable for simulation environments where fleet size is flexible.
    - strict_fleet=True (Benchmark Mode): Respect fixed fleet size K.
      Raises ValueError if greedy heuristic fails to assign all nodes to K vehicles.
      This mode is required for standard CVRP benchmark comparisons (e.g., A-VRP dataset).

    Args:
        seeds: List of seed node indices (one per initial cluster).
        must_go: List of all nodes that need to be assigned.
        wastes: Dictionary mapping node indices to waste quantities.
        capacity: Maximum vehicle capacity.
        distance_matrix: Pre-computed all-pairs distance matrix.
        strict_fleet: If True, raise error for unassigned nodes instead of opening new vehicles.

    Returns:
        List of clusters, where each cluster is a list of node indices.

    Raises:
        ValueError: If strict_fleet=True and greedy heuristic cannot assign all nodes to K vehicles.

    Example:
        >>> seeds = [5, 12, 20]
        >>> must_go = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        >>> wastes = {i: 10.0 for i in must_go}
        >>> capacity = 100.0
        >>> distance_matrix = np.random.rand(21, 21)
        >>> # Flexible mode (default)
        >>> clusters = assign_greedy(seeds, must_go, wastes, capacity, distance_matrix)
        >>> len(clusters) >= len(seeds)  # May open new vehicles if needed
        True
        >>> # Strict benchmark mode
        >>> clusters = assign_greedy(seeds, must_go, wastes, capacity, distance_matrix, strict_fleet=True)
        >>> len(clusters) == len(seeds)  # Exactly K vehicles
        True
    """
    # Initialize clusters and track capacity usage
    clusters: List[List[int]] = [[] for _ in range(len(seeds))]
    loads = [0.0] * len(seeds)
    assigned = set()

    # Step 1: Pre-assign seeds to their own clusters
    for k_idx, k in enumerate(seeds):
        if k in must_go:
            clusters[k_idx].append(k)
            loads[k_idx] += wastes.get(k, 0.0)
            assigned.add(k)

    # Step 2: Calculate insertion costs for all (node, seed) pairs
    insertion_costs = []
    for i in must_go:
        if i in assigned:
            continue  # Skip already assigned seeds
        for k_idx, k in enumerate(seeds):
            # Fisher & Jaikumar insertion cost: C_ik = d(0, i) + d(k, i) - d(k, 0)
            cost = distance_matrix[0, i] + distance_matrix[k, i] - distance_matrix[k, 0]
            insertion_costs.append((cost, i, k_idx))

    # Step 3: Sort by increasing insertion cost (greedy selection)
    # Use node_i as secondary sort key for deterministic tie-breaking
    insertion_costs.sort(key=lambda x: (x[0], x[1]))

    # Step 4: Assign nodes greedily
    for _cost, node_i, seed_k_idx in insertion_costs:
        if node_i in assigned:
            continue  # Node already assigned

        waste_i = wastes.get(node_i, 0.0)

        # Check capacity constraint
        if loads[seed_k_idx] + waste_i <= capacity:
            clusters[seed_k_idx].append(node_i)
            loads[seed_k_idx] += waste_i
            assigned.add(node_i)

    # Step 5: Safety net - handle unassigned nodes
    # Behavior depends on strict_fleet mode
    unassigned = [node for node in must_go if node not in assigned]

    if not unassigned:
        # All nodes successfully assigned
        return clusters

    # Try to fit unassigned nodes into existing clusters with spare capacity
    for node in unassigned:
        waste = wastes.get(node, 0.0)

        fitted = False
        for k_idx in range(len(seeds)):
            if loads[k_idx] + waste <= capacity:
                clusters[k_idx].append(node)
                loads[k_idx] += waste
                fitted = True
                assigned.add(node)
                break

        if not fitted:
            # Node cannot fit into any existing cluster
            if strict_fleet:
                # STRICT MODE: Raise error (benchmark compliance)
                # This indicates the greedy heuristic failed for the given K
                unassigned_final = [n for n in must_go if n not in assigned]
                raise ValueError(
                    f"Greedy assignment failed in strict fleet mode. "
                    f"Cannot assign {len(unassigned_final)} nodes to {len(seeds)} vehicles. "
                    f"Unassigned nodes: {unassigned_final}. "
                    f"Consider: (1) increasing fleet size K, (2) increasing vehicle capacity, "
                    f"or (3) using exact MIP assignment instead of greedy."
                )
            else:
                # FLEXIBLE MODE: Open new vehicle (simulation mode)
                seeds.append(node)
                clusters.append([node])
                loads.append(waste)
                assigned.add(node)

    return clusters
