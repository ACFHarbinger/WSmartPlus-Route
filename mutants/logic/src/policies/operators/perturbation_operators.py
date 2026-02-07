"""
Perturbation operators for escaping local optima.
"""

import random
from typing import Any


def perturb(ctx: Any, k: int = 3) -> bool:
    """
    Perturbation operator: perform k random swaps to escape local optima.
    Returns True if any swap was performed.

    Args:
        ctx: Context object with routes, node_map, demands, capacity, etc.
        k: Number of swaps to perform.
    """
    all_nodes = list(ctx.node_map.keys())
    if len(all_nodes) < 2:
        return False

    performed = False
    for _ in range(k):
        if len(all_nodes) < 2:
            break
        u, v = random.sample(all_nodes, 2)
        u_loc = ctx.node_map.get(u)
        v_loc = ctx.node_map.get(v)
        if not u_loc or not v_loc:
            continue

        r_u, p_u = u_loc
        r_v, p_v = v_loc

        # Force swap regardless of improvement
        if r_u == r_v:
            # Same route swap
            if abs(p_u - p_v) > 1:
                ctx.routes[r_u][p_u], ctx.routes[r_v][p_v] = (
                    ctx.routes[r_v][p_v],
                    ctx.routes[r_u][p_u],
                )
                ctx._update_map({r_u})
                performed = True
        else:
            # Inter-route swap
            dem_u = ctx.demands.get(u, 0)
            dem_v = ctx.demands.get(v, 0)
            if (
                ctx._get_load_cached(r_u) - dem_u + dem_v <= ctx.Q
                and ctx._get_load_cached(r_v) - dem_v + dem_u <= ctx.Q
            ):
                ctx.routes[r_u][p_u] = v
                ctx.routes[r_v][p_v] = u
                ctx._update_map({r_u, r_v})
                performed = True

    return performed


def kick(ctx: Any, destroy_ratio: float = 0.2) -> bool:
    """
    Kick operator: destroy a portion of the solution and repair.
    Removes random nodes and reinserts them greedily.
    Returns True if the operation was performed.

    Args:
        ctx: Context object with routes, node_map, demands, capacity, etc.
        destroy_ratio: Fraction of nodes to remove.
    """
    all_nodes = list(ctx.node_map.keys())
    n_remove = max(1, int(len(all_nodes) * destroy_ratio))
    if len(all_nodes) < 2:
        return False

    # Select nodes to remove
    removed_nodes = random.sample(all_nodes, min(n_remove, len(all_nodes)))

    # Remove nodes from routes
    for node in removed_nodes:
        loc = ctx.node_map.get(node)
        if loc:
            ri, pi = loc
            if pi < len(ctx.routes[ri]) and ctx.routes[ri][pi] == node:
                ctx.routes[ri].pop(pi)

    # Remove empty routes
    ctx.routes = [r for r in ctx.routes if r]

    # Rebuild structures after removal
    ctx._build_structures()

    # Greedy reinsertion
    for node in removed_nodes:
        best_cost = float("inf")
        best_ri = -1
        best_pos = -1

        for ri, route in enumerate(ctx.routes):
            current_load = ctx._get_load_cached(ri)
            node_demand = ctx.demands.get(node, 0)
            if current_load + node_demand > ctx.Q:
                continue

            # Try inserting at each position
            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0
                cost = ctx.d[prev, node] + ctx.d[node, nxt] - ctx.d[prev, nxt]
                if cost < best_cost:
                    best_cost = cost
                    best_ri = ri
                    best_pos = pos

        if best_ri >= 0:
            ctx.routes[best_ri].insert(best_pos, node)
        else:
            # Create new route
            ctx.routes.append([node])

    # Rebuild structures after reinsertion
    ctx._build_structures()
    return True
