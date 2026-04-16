"""
Inter-Period Destroy Operators for Multi-Period ALNS.

Implements horizon-aware removal operators that operate on the full T-day
chromosome ``horizon_routes: List[List[List[int]]]``, enabling ALNS to
reason across days rather than myopically optimising a single period.

Operators
---------
shift_visit_removal
    Removes a visit to a node on day ``t`` and marks it for re-insertion
    on an adjacent day.  This is the primary driver of schedule reshaping:
    it allows the ALNS to slide visits earlier or later when inventory
    dynamics make that more profitable.

pattern_removal
    Erases the entire visit-frequency pattern for a node, removing it from
    every day in the horizon.  This enables a wholesale re-scheduling of a
    node and is analogous to Coelho et al. (2012) "path removal" extended
    to the inventory routing literature.

References
----------
Coelho, L. C., Cordeau, J.-F., & Laporte, G. (2012). "The inventory-routing
problem with transshipment." Computers & Operations Research, 39(11), 2537–2548.
"""

from __future__ import annotations

import random as _random
from typing import Dict, List, Optional, Set, Tuple


def shift_visit_removal(  # noqa: C901
    horizon_routes: List[List[List[int]]],
    n_remove: int,
    direction: str = "both",
    wastes: Optional[Dict[int, float]] = None,
    rng: Optional[_random.Random] = None,
) -> Tuple[List[List[List[int]]], List[Tuple[int, int]]]:
    """Remove visits and mark them for insertion on an adjacent day.

    For each selected (node, day) pair the node is removed from
    ``horizon_routes[t]`` and recorded as ``(node, t_target)`` where
    ``t_target`` is the adjacent day chosen according to ``direction``.
    The ALNS repair operator is then responsible for re-inserting the node
    on ``t_target``.

    Args:
        horizon_routes: Full T-day solution ``[day][route][node]``.
        n_remove: Number of (node, day) pairs to remove.
        direction: Which adjacent day to target for re-insertion.
            ``"forward"`` → always day ``t+1``.
            ``"backward"`` → always day ``t-1``.
            ``"both"`` → randomly choose between ``t-1`` and ``t+1``.
        wastes: Optional mapping ``{node_id: fill_level}`` used to bias
            removal toward high-fill nodes (more urgent shifts).
        rng: Random number generator for reproducibility.

    Returns:
        Tuple of:
            - Modified ``horizon_routes`` with selected visits removed.
            - List of ``(node_id, target_day)`` pairs for re-insertion.
    """
    if rng is None:
        rng = _random.Random()

    T = len(horizon_routes)
    if T == 0:
        return horizon_routes, []

    # Build flat list of (node, day) candidates
    candidates: List[Tuple[int, int]] = []
    for t, day_routes in enumerate(horizon_routes):
        for route in day_routes:
            for node in route:
                candidates.append((node, t))

    if not candidates:
        return horizon_routes, []

    n_remove = min(n_remove, len(candidates))

    # Optional bias: weight by fill level (higher fill → higher removal priority)
    if wastes and all(c[0] in wastes for c in candidates):
        fills = [wastes[node] for node, _ in candidates]
        max_fill = max(fills) if fills else 1.0
        min_fill = min(fills) if fills else 0.0
        fill_range = max_fill - min_fill if max_fill != min_fill else 1.0
        weights = [(f - min_fill) / fill_range + 0.1 for f in fills]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        selected: List[Tuple[int, int]] = []
        selected_set: Set[int] = set()
        for _ in range(n_remove * 10):
            idx = rng.choices(range(len(candidates)), weights=probs, k=1)[0]
            if idx not in selected_set:
                selected.append(candidates[idx])
                selected_set.add(idx)
            if len(selected) >= n_remove:
                break
    else:
        selected = rng.sample(candidates, n_remove)

    # Determine target days
    removed_with_target: List[Tuple[int, int]] = []
    set(selected)

    for node, t in selected:
        if direction == "forward":
            t_target = min(t + 1, T - 1)
        elif direction == "backward":
            t_target = max(t - 1, 0)
        else:  # "both"
            candidates_t: List[int] = []
            if t > 0:
                candidates_t.append(t - 1)
            if t < T - 1:
                candidates_t.append(t + 1)
            t_target = rng.choice(candidates_t) if candidates_t else t
        removed_with_target.append((node, t_target))

    # Remove selected nodes from their source days
    removed_pairs: Set[Tuple[int, int]] = {(node, t) for node, t in selected}
    new_horizon: List[List[List[int]]] = []
    for t, day_routes in enumerate(horizon_routes):
        new_day: List[List[int]] = []
        for route in day_routes:
            new_route = [node for node in route if (node, t) not in removed_pairs]
            if new_route:
                new_day.append(new_route)
        new_horizon.append(new_day)

    return new_horizon, removed_with_target


def pattern_removal(
    horizon_routes: List[List[List[int]]],
    n_remove: int,
    wastes: Optional[Dict[int, float]] = None,
    rng: Optional[_random.Random] = None,
) -> Tuple[List[List[List[int]]], List[Tuple[int, int]]]:
    """Remove the complete visit pattern for a node across the horizon.

    Identifies the set of unique nodes currently in the horizon and
    removes all occurrences of ``n_remove`` selected nodes from every day.
    The removed visits are encoded as ``(node, t)`` pairs for all days
    where the node was scheduled, enabling the repair operator to
    reconstruct the full service pattern.

    Args:
        horizon_routes: Full T-day solution ``[day][route][node]``.
        n_remove: Number of distinct nodes whose patterns to erase.
        wastes: Optional fill-level mapping for biasing toward urgent nodes.
        rng: Random number generator.

    Returns:
        Tuple of:
            - Modified ``horizon_routes`` with all visits to selected
              nodes removed across all days.
            - List of ``(node_id, day_index)`` pairs recording every
              erased visit.
    """
    if rng is None:
        rng = _random.Random()

    T = len(horizon_routes)
    if T == 0:
        return horizon_routes, []

    # Collect unique nodes present in the horizon
    node_day_occurrences: Dict[int, List[int]] = {}
    for t, day_routes in enumerate(horizon_routes):
        for route in day_routes:
            for node in route:
                node_day_occurrences.setdefault(node, []).append(t)

    unique_nodes = list(node_day_occurrences.keys())
    if not unique_nodes:
        return horizon_routes, []

    n_remove = min(n_remove, len(unique_nodes))

    # Optional bias toward high-fill nodes
    if wastes:
        fills = [wastes.get(node, 0.0) for node in unique_nodes]
        max_fill = max(fills) if fills else 1.0
        min_fill = min(fills) if fills else 0.0
        fill_range = max_fill - min_fill if max_fill != min_fill else 1.0
        weights = [(f - min_fill) / fill_range + 0.1 for f in fills]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        chosen_nodes = rng.choices(unique_nodes, weights=probs, k=n_remove)
        # De-duplicate while preserving order
        seen: Set[int] = set()
        chosen_nodes_dedup: List[int] = []
        for n in chosen_nodes:
            if n not in seen:
                chosen_nodes_dedup.append(n)
                seen.add(n)
        chosen_nodes = chosen_nodes_dedup[:n_remove]
    else:
        chosen_nodes = rng.sample(unique_nodes, n_remove)

    chosen_set: Set[int] = set(chosen_nodes)

    # Build removed list: all (node, day) occurrences for chosen nodes
    removed: List[Tuple[int, int]] = [
        (node, t) for node, days in node_day_occurrences.items() if node in chosen_set for t in days
    ]

    # Remove from horizon
    new_horizon: List[List[List[int]]] = []
    for _t, day_routes in enumerate(horizon_routes):
        new_day: List[List[int]] = []
        for route in day_routes:
            new_route = [node for node in route if node not in chosen_set]
            if new_route:
                new_day.append(new_route)
        new_horizon.append(new_day)

    return new_horizon, removed
