"""
Local Search Base Module.

This module defines the abstract base class for local search algorithms.
It handles common initialization, neighbor lists, and provides wrappers
for atomic move operators.

Attributes:
    None

Example:
    >>> # Cannot be instantiated directly
    >>> class MyLS(LocalSearch): ...
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..operators.inter_route import (
    move_2opt_star,
    move_swap_star,
)
from ..operators.inter_route.cross_exchange import cross_exchange, improved_cross_exchange, lambda_interchange
from ..operators.inter_route.cyclic_transfer import cyclic_transfer
from ..operators.inter_route.ejection_chain import ejection_chain
from ..operators.inter_route.exchange_chain import exchange_2_0, exchange_2_1
from ..operators.intra_route import (
    move_2opt_intra,
    move_3opt_intra,
    move_or_opt,
    move_relocate,
    move_swap,
    relocate_chain,
)
from ..operators.intra_route.k_permutation import three_permutation


class LocalSearch(ABC):
    """
    Abstract base class for Local Search algorithms.
    Provides common infrastructure for neighbor lists and move operators.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
        neighbors: Optional[Dict[int, List[int]]] = None,
        penalty_capacity: float = 1.0,
    ):
        """
        Initialize Local Search base class.

        Args:
            dist_matrix: The distance matrix between nodes.
            waste: A dictionary mapping node IDs to their waste amounts.
            capacity: The maximum capacity of a vehicle.
            R: A parameter, likely related to route cost or penalty.
            C: A parameter, likely related to route cost or penalty.
            params: An object containing additional parameters for the local search,
                    such as time_limit.
            neighbors: Optional pre-computed neighbor map to avoid redundant sorting.
            penalty_capacity: Penalty coefficient for capacity violations (Vidal 2022).
        """
        self.d = np.array(dist_matrix)
        self.waste = waste
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.penalty_capacity = penalty_capacity
        self.random = random.Random(params.seed) if params.seed is not None else random.Random(42)

        if neighbors is not None:
            self.neighbors = neighbors
        else:
            # Common initialization for neighbors (used by all LS)
            n_nodes = len(dist_matrix)
            self.neighbors = {}
            for i in range(1, n_nodes):
                row = self.d[i]
                order = np.argsort(row)
                cands = []
                for c in order:
                    if c not in (i, 0):
                        cands.append(c)
                        if len(cands) >= 10:
                            break
                self.neighbors[i] = cands

        self.node_map: Dict[int, Tuple[int, int]] = {}
        self.route_loads: List[float] = []
        self.routes: List[List[int]] = []

        # Top-3 Insertion Cache for O(1) SWAP* evaluation (Vidal 2022)
        # Maps: node_id -> route_idx -> [(cost_delta, position), ...]
        # Stores the 3 best insertion positions for each node into each route
        self.top_insertions: Dict[int, Dict[int, List[Tuple[float, int]]]] = {}

    @abstractmethod
    def optimize(self, solution: Any) -> Any:
        """
        Optimize the given solution.
        """
        pass

    def _optimize_internal(
        self,
        target_neighborhood: Optional[str] = None,
        active_nodes: Optional[Set[int]] = None,
    ):
        """
        Core local search loop. Assumes self.routes is populated.

        Args:
            target_neighborhood: If provided, only applies the specified neighborhood operator.
                               If None or "all", applies all available operators.
            active_nodes: If provided, restricts search to moves involving at least one active node.
                        This enables O(1) localization per iteration (FILO).
        """
        self.route_loads = [self._calc_load_fresh(r) for r in self.routes]

        # Initialize node map
        self.node_map.clear()
        for ri, r in enumerate(self.routes):
            for pi, node in enumerate(r):
                self.node_map[node] = (ri, pi)

        # Initialize Top-3 Insertion Cache for O(1) SWAP* evaluation (Vidal 2022)
        self._compute_top_insertions()

        # Store target neighborhood for _process_node to filter operators
        self._target_neighborhood = target_neighborhood if target_neighborhood else "all"

        # Store active nodes set for localization (FILO)
        self._active_nodes = active_nodes

        improved = True
        it = 0
        t_start = time.process_time()
        while improved and it < self.params.local_search_iterations:
            if self.params.time_limit > 0 and time.process_time() - t_start > self.params.time_limit:
                break

            improved = False
            it += 1

            # Include ALL nodes (visited + unvisited) for VRPP local search
            # This allows re-insertion of nodes dropped by Split algorithm
            nodes = list(self.neighbors.keys())
            self.random.shuffle(nodes)

            for u in nodes:
                if self._process_node(u):
                    improved = True
                    break

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=it,
                n_routes=len(self.routes),
                improved=int(improved),
            )

    def _calc_load_fresh(self, r: List[int]) -> float:
        return sum(self.waste.get(x, 0) for x in r)

    def _compute_top_insertions(self, route_idx: Optional[int] = None):
        """
        Compute or update the Top-3 Insertion Cache for O(1) SWAP* evaluation.

        Following Vidal et al. (2022), this cache stores the 3 best insertion positions
        for each node into each route, enabling constant-time insertion cost evaluation.

        Args:
            route_idx: If provided, update cache only for this route. If None, initialize for all routes.
        """
        routes_to_update = [route_idx] if route_idx is not None else range(len(self.routes))

        for r_idx in routes_to_update:
            route = self.routes[r_idx]
            if not route:
                continue

            # For each node that could potentially be inserted into this route
            for node in self.neighbors:
                # Skip nodes already in this route
                if node in route:
                    continue

                # Initialize cache structure if needed
                if node not in self.top_insertions:
                    self.top_insertions[node] = {}

                # Calculate insertion costs for all positions
                insertion_costs = []
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    delta = self.d[prev, node] + self.d[node, nxt] - self.d[prev, nxt]
                    insertion_costs.append((delta, pos))

                # Keep top 3 (lowest cost) insertions
                insertion_costs.sort(key=lambda x: x[0])
                self.top_insertions[node][r_idx] = insertion_costs[:3]

    def _should_try_operator(self, op_name: str) -> bool:
        """Check if the operator should be tried based on target_neighborhood filter."""
        target = getattr(self, "_target_neighborhood", "all")
        return target == "all" or target == op_name

    def _process_node(self, u: int) -> bool:  # noqa: C901
        """
        Process node u with all neighbors, handling both visited and unvisited nodes.

        For VRPP, this method handles four scenarios:
        - Both nodes unvisited: skip (no routing context)
        - u unvisited, v visited: try inserting u into v's route
        - u visited, v unvisited: try inserting v into u's route
        - Both visited: standard inter/intra route operators

        Operators are filtered based on self._target_neighborhood if set.
        Active node localization (FILO) is enforced via self._active_nodes if set.
        """
        u_loc = self.node_map.get(u)

        for v in self.neighbors[u]:
            # FILO LOCALIZATION GUARD: O(1) per iteration
            # Skip moves that don't involve at least one active node
            if self._active_nodes is not None and u not in self._active_nodes and v not in self._active_nodes:
                continue
            v_loc = self.node_map.get(v)

            # Scenario A: Both are unvisited -> skip, no routing context
            if not u_loc and not v_loc:
                continue

            # Scenario B: u is unvisited, v is visited
            # Action: Try inserting unrouted node u into v's route
            if not u_loc and v_loc:
                if self._should_try_operator("unrouted_insert"):
                    r_v, p_v = v_loc
                    if self._move_unrouted_insert(u, r_v, p_v):
                        return True
                continue

            # Scenario C: u is visited, v is unvisited
            # Action: Try inserting unrouted node v into u's route
            if u_loc and not v_loc:
                if self._should_try_operator("unrouted_insert"):
                    r_u, p_u = u_loc
                    if self._move_unrouted_insert(v, r_u, p_u):
                        return True
                continue

            # Scenario D: Both are visited (Standard HGS inter/intra route operators)
            r_u, p_u = u_loc  # type: ignore[misc]
            r_v, p_v = v_loc  # type: ignore[misc]

            # Relocate (both intra and inter-route)
            if (
                self._should_try_operator("relocate")
                or self._should_try_operator("intra_relocate" if r_u == r_v else "inter_relocate")
            ) and self._move_relocate(u, v, r_u, p_u, r_v, p_v):
                return True

            # Swap (both intra and inter-route)
            if (
                self._should_try_operator("swap")
                or self._should_try_operator("intra_swap" if r_u == r_v else "inter_swap")
            ) and self._move_swap(u, v, r_u, p_u, r_v, p_v):
                return True

            # Relocate Chain
            if (
                self._should_try_operator("relocate_chain")
                and getattr(self.params, "use_relocate_chain", False)
                and self._move_relocate_chain(u, r_u, p_u, r_v, p_v)
            ):
                return True

            # Inter-route operators
            if r_u != r_v:
                if self._should_try_operator("inter_2opt_star") and self._move_2opt_star(u, v, r_u, p_u, r_v, p_v):
                    return True

                if self._should_try_operator("inter_swap_star") and self._move_swap_star(u, v, r_u, p_u, r_v, p_v):
                    return True

                # Advanced inter-route operators
                if (
                    self._should_try_operator("cross_exchange")
                    and getattr(self.params, "use_cross_exchange", False)
                    and self._try_cross_exchange(r_u, p_u, r_v, p_v)
                ):
                    return True

                if (
                    self._should_try_operator("improved_cross_exchange")
                    and getattr(self.params, "use_improved_cross_exchange", False)
                    and self._try_improved_cross_exchange(r_u, p_u, r_v, p_v)
                ):
                    return True

                if (
                    self._should_try_operator("lambda_interchange")
                    and getattr(self.params, "use_lambda_interchange", False)
                    and self._try_lambda_interchange(r_u, r_v)
                ):
                    return True

                if (
                    self._should_try_operator("cyclic_transfer")
                    and getattr(self.params, "use_cyclic_transfer", False)
                    and self._try_cyclic_transfer(r_u, p_u, r_v, p_v)
                ):
                    return True

                if (
                    self._should_try_operator("exchange_chains")
                    and getattr(self.params, "use_exchange_chains", False)
                    and self._try_exchange_chains(r_u, p_u, r_v, p_v)
                ):
                    return True

                if (
                    self._should_try_operator("ejection_chains")
                    and getattr(self.params, "use_ejection_chains", False)
                    and self._try_ejection_chain(r_u)
                ):
                    return True
            else:
                if self._should_try_operator("intra_2opt") and self._move_2opt_intra(u, v, r_u, p_u, r_v, p_v):
                    return True

                if self._should_try_operator("intra_3opt") and self._move_3opt_intra(
                    u, v, r_u, p_u, r_v, p_v, self.random
                ):
                    return True

                if self._should_try_operator("intra_or_opt") and self._move_or_opt(
                    r_u, p_u, self.random.choice([1, 2, 3])
                ):
                    return True

                if (
                    self._should_try_operator("three_permutation")
                    and getattr(self.params, "use_three_permutation", False)
                    and self._move_three_permutation(u, r_u, p_u)
                ):
                    return True

        return False

    def _update_map(self, affected_indices: Set[int]):
        """
        Update node map and route loads after route modifications.
        Also updates Top-3 Insertion Cache for affected routes (Vidal 2022).

        Args:
            affected_indices: Set of route indices that were modified.
        """
        for ri in sorted(affected_indices):
            for pi, node in enumerate(self.routes[ri]):
                self.node_map[node] = (ri, pi)
            self.route_loads[ri] = self._calc_load_fresh(self.routes[ri])
            # Update Top-3 cache for this modified route (O(1) SWAP* maintenance)
            self._compute_top_insertions(route_idx=ri)

    def _get_load_cached(self, ri: int) -> float:
        return self.route_loads[ri]

    def _move_relocate(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_relocate(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_swap(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_swap_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_3opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, rng: random.Random) -> bool:
        return move_3opt_intra(self, u, v, r_u, p_u, r_v, p_v, rng)

    def _move_2opt_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_2opt_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_2opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_2opt_intra(self, u, v, r_u, p_u, r_v, p_v)

    def _move_or_opt(self, r_idx: int, pos: int, chain_len: int) -> bool:
        return move_or_opt(self, r_idx, pos, chain_len)

    def _try_cross_exchange(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return cross_exchange(self, r_u, p_u, 1, r_v, p_v, 1)

    def _try_improved_cross_exchange(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return improved_cross_exchange(self, r_u, p_u, 1, r_v, p_v, 1)

    def _try_lambda_interchange(self, r_u: int, r_v: int) -> bool:
        return lambda_interchange(self, getattr(self.params, "lambda_max", 2))

    def _try_cyclic_transfer(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        if len(self.routes) < 3:
            return False
        candidates = [r for r in range(len(self.routes)) if r != r_u and r != r_v and self.routes[r]]
        if not candidates:
            return False
        r_w = self.random.choice(candidates)
        p_w = self.random.randint(0, len(self.routes[r_w]) - 1)
        return cyclic_transfer(self, [(r_u, p_u), (r_v, p_v), (r_w, p_w)])

    def _try_exchange_chains(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        if exchange_2_0(self, r_u, p_u, r_v, p_v):
            return True
        return bool(exchange_2_1(self, r_u, p_u, r_v, p_v))

    def _try_ejection_chain(self, r_u: int) -> bool:
        return ejection_chain(self, r_u)

    def _move_relocate_chain(self, u: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        if relocate_chain(self, r_u, p_u, r_v, p_v, chain_len=2):
            return True
        return bool(relocate_chain(self, r_u, p_u, r_v, p_v, chain_len=3))

    def _move_three_permutation(self, u: int, r_u: int, p_u: int) -> bool:
        return three_permutation(self, r_u, p_u)

    def _move_unrouted_insert(self, node: int, route_idx: int, _hint_pos: int) -> bool:
        """
        Try inserting an unrouted node back into a route in O(1) time (VRPP operator).

        Evaluates inserting 'node' (currently unvisited) into 'route_idx' at the best position.
        Uses Top-3 Insertion Cache for O(1) position lookup instead of O(N) loop.
        Uses penalized search: considers both profit gain and capacity penalties.

        Args:
            node: Unvisited node to potentially insert.
            route_idx: Route to insert into.
            _hint_pos: Position hint (near a neighbor), currently unused.

        Returns:
            bool: True if insertion improves penalized profit, False otherwise.
        """
        route = self.routes[route_idx]

        # 1. Base Case: Empty route
        if not route:
            cost_delta = 2 * self.d[0, node] * self.C
            best_pos = 0
        else:
            # 2. O(1) Cache Query - No loops required!
            # Since 'node' is unrouted, all cached positions are valid (no removal offsets)
            if node in self.top_insertions and route_idx in self.top_insertions[node]:
                # Always use first cached position (guaranteed valid for unrouted nodes)
                best_delta_raw, best_pos = self.top_insertions[node][route_idx][0]
                cost_delta = best_delta_raw * self.C
            else:
                # Fallback safety if cache is missing (shouldn't happen with proper maintenance)
                return False

        profit_delta = self.waste.get(node, 0) * self.R
        load_delta = self.waste.get(node, 0)

        # 3. Penalties & Evaluation (Standard HGS logic)
        penalty_weight = getattr(self, "penalty_capacity", 1.0)
        current_load = self._get_load_cached(route_idx)
        new_load = current_load + load_delta

        old_penalty = penalty_weight * max(0.0, current_load - self.Q)
        new_penalty = penalty_weight * max(0.0, new_load - self.Q)
        penalty_delta = new_penalty - old_penalty

        # Total change in objective: profit gain - cost increase - penalty increase
        total_change = profit_delta - cost_delta - penalty_delta

        # Accept if improves penalized profit
        if total_change > 1e-4:
            if not route:
                self.routes[route_idx] = [node]
            else:
                self.routes[route_idx].insert(best_pos, node)
            self._update_map({route_idx})
            return True

        return False
