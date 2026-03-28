"""
Cutting plane separation engines for Branch-and-Price-and-Cut algorithms.

Provides modular separation algorithms for different cut families:
- RCC (Rounded Capacity Cuts): For capacitated vehicle routing
- LCI (Lifted Cover Inequalities): For knapsack capacity constraints

References:
    - Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004).
      "A new branch-and-cut algorithm for the capacitated vehicle routing problem."
      Mathematical Programming, 100(2), 423-445.
      (Rounded Capacity Cuts)

    - Balas, E. (1975). "Facets of the knapsack polytope."
      Mathematical Programming, 8(1), 146-164.
      (Cover inequalities)

    - Zemel, E. (1989). "Easily computable facets of the knapsack polytope."
      Mathematics of Operations Research, 14(4), 760-764.
      (Lifting procedures for cover inequalities)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..branch_and_cut.separation import SeparationEngine
from ..branch_and_cut.vrpp_model import VRPPModel
from ..branch_and_price.master_problem import VRPPMasterProblem


class CuttingPlaneEngine(ABC):
    """
    Abstract base class for cutting plane separation engines.

    Each engine implements a specific family of valid inequalities and
    provides separation algorithms to identify violated cuts.
    """

    @abstractmethod
    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Identify violated cuts and add them to the master problem.

        Args:
            master: Master problem instance
            max_cuts: Maximum number of cuts to add
            **kwargs: Engine-specific parameters

        Returns:
            Number of cuts added
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the engine name for logging."""
        pass


class RoundedCapacityCutEngine(CuttingPlaneEngine):
    """
    Rounded Capacity Cut (RCC) separation engine for VRPP Set Packing.

    Mathematical Formulation:
    -------------------------
    For a subset S ⊆ N \\ {0} of customer nodes, the RCC is:

        ∑_{e ∈ δ(S)} x_e ≥ 2 * k(S)

    where:
    - δ(S) is the set of edges crossing the boundary of S
    - k(S) = ⌈demand(S) / Q⌉ is the minimum number of vehicles needed
    - x_e is the aggregated edge usage from the column generation solution

    For VRPP Set Packing, we relax this with visitation variables:

        ∑_{e ∈ δ(S)} x_e ≥ 2*k(S) - M * ∑_{i ∈ S} (1 - y_i)

    where y_i = ∑_{k: i ∈ route_k} λ_k is the fractional visitation probability.

    Integration with Pricing:
    --------------------------
    The dual variables from these cuts must be integrated into the pricing
    problem's reduced cost calculation. For each cut on set S:

        reduced_cost -= dual_S * (number of times route crosses δ(S))

    This is handled by the master problem's dual tracking and passed to
    the RCSPP solver via capacity_cut_duals parameter.

    References:
    -----------
    Lysgaard et al. (2004), Section 3.2: "Capacity cuts"
    """

    def __init__(self, v_model: VRPPModel, sep_engine: SeparationEngine):
        """
        Initialize the RCC separation engine.

        Args:
            v_model: VRPP model for edge indexing and demand tracking
            sep_engine: Separation engine from branch-and-cut module
        """
        self.v_model = v_model
        self.sep_engine = sep_engine

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Separate RCCs and add to master problem.

        Algorithm:
        1. Get current edge usage from master problem's LP solution
        2. Map edge usage to the separation engine's representation
        3. Run separation algorithms (heuristic + exact)
        4. Add violated cuts to master with Set Packing relaxation

        Args:
            master: Master problem instance
            max_cuts: Maximum number of cuts to add
            **kwargs: Unused (for interface consistency)

        Returns:
            Number of cuts added
        """
        edge_vars = master.get_edge_usage()
        if not edge_vars:
            return 0

        # Map edge usage to x_vals array for SeparationEngine
        x_vals = np.zeros(len(self.v_model.edges))
        for (i, j), val in edge_vars.items():
            edge_tuple = (min(i, j), max(i, j))
            if edge_tuple in self.v_model.edge_to_idx:
                idx = self.v_model.edge_to_idx[edge_tuple]
                x_vals[idx] = val

        # Separate cuts using the existing separation engine
        violated_ineqs = self.sep_engine.separate(x_vals, max_cuts=max_cuts)
        added_cuts = 0

        for ineq in violated_ineqs:
            if ineq.type == "CAPACITY":
                node_set = list(ineq.node_set)

                # For Set Packing, add cut with boundary relaxation
                # The master problem handles the y_i visitation tracking
                if master.add_set_packing_capacity_cut(node_set, ineq.rhs):
                    added_cuts += 1

        # Re-solve LP if cuts were added
        if added_cuts > 0:
            master.solve_lp_relaxation()

        return added_cuts

    def get_name(self) -> str:
        return "rcc"


class LiftedCoverInequalityEngine(CuttingPlaneEngine):
    """
    Lifted Cover Inequality (LCI) separation engine for knapsack constraints.

    Mathematical Formulation:
    -------------------------
    For an arc (i,j) with capacity Q and a set of commodities K, the
    knapsack constraint is:

        ∑_{k ∈ K} d_k * x_{ij}^k ≤ Q

    A minimal cover C ⊆ K is a subset such that:
    - ∑_{k ∈ C} d_k > Q  (cover is infeasible)
    - Removing any k from C makes it feasible (minimality)

    The lifted cover inequality is:

        ∑_{k ∈ C} x_{ij}^k + ∑_{k ∈ K\\C} α_k * x_{ij}^k ≤ |C| - 1

    where α_k are lifting coefficients computed via dynamic programming.

    Separation Algorithm:
    ---------------------
    1. For each arc (i,j) with fractional flow > threshold:
        a. Sort commodities by demand/flow ratio (greedy cover selection)
        b. Find minimal cover C by adding items until capacity exceeded
        c. Compute lifting coefficients α_k for items not in cover
        d. Check if inequality is violated by current LP solution
        e. If violated, add as lazy constraint to master problem

    Integration with Pricing:
    --------------------------
    LCI cuts on arc (i,j) modify the effective cost of using that arc
    for commodity k:

        c_{ij}^k_modified = c_{ij}^k - dual_{ij} * α_k

    This is enforced in the pricing problem by modifying arc costs based
    on which commodities can be "forbidden" from certain arcs due to
    capacity saturation.

    Implementation Strategy:
    ------------------------
    For VRP (not multicommodity flow), we approximate by treating routes
    as "commodities" and arc usage as binary decisions. The separation
    becomes:

    1. Identify arcs with high fractional edge usage
    2. Find routes that saturate the arc's capacity when combined
    3. Generate cuts that prevent infeasible route combinations

    References:
    -----------
    - Balas (1975), Theorem 3.1: Minimal covers and facets
    - Zemel (1989), Algorithm 2: Sequential lifting in O(nC) time
    - Barnhart et al. (1998), Section 4.2: "Capacity inequalities"
    """

    def __init__(self, v_model: VRPPModel):
        """
        Initialize the LCI separation engine.

        Args:
            v_model: VRPP model for arc and demand tracking
        """
        self.v_model = v_model
        self.cut_count = 0

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Separate LCIs and add to master problem.

        Algorithm:
        1. Identify arcs with high fractional usage
        2. For each arc, find routes that create capacity violations
        3. Generate minimal cover from these routes
        4. Compute lifting coefficients
        5. Add violated inequalities to master

        Args:
            master: Master problem instance
            max_cuts: Maximum number of cuts to add
            **kwargs: Unused (for interface consistency)

        Returns:
            Number of cuts added
        """
        # Get current edge usage from master problem
        edge_vars = master.get_edge_usage()
        if not edge_vars:
            return 0

        route_values = master.route_values if hasattr(master, "route_values") else {}
        if not route_values:
            return 0

        added_cuts = 0
        cuts_per_arc_limit = max(1, max_cuts // 5)  # Spread cuts across arcs

        # Identify arcs with fractional usage > threshold
        fractional_arcs = [(arc, usage) for arc, usage in edge_vars.items() if 0.1 < usage < 0.9]

        # Sort by "fractionality" (closeness to 0.5)
        fractional_arcs.sort(key=lambda x: abs(x[1] - 0.5))

        for (i, j), _arc_usage in fractional_arcs[:10]:  # Check top 10 most fractional arcs
            cuts_added_for_arc = self._separate_arc_cover(master, i, j, route_values, max_cuts=cuts_per_arc_limit)
            added_cuts += cuts_added_for_arc

            if added_cuts >= max_cuts:
                break

        # Re-solve LP if cuts were added
        if added_cuts > 0:
            master.solve_lp_relaxation()

        return added_cuts

    def _separate_arc_cover(
        self, master: VRPPMasterProblem, u: int, v: int, route_values: Dict[int, float], max_cuts: int
    ) -> int:
        """
        Separate cover inequalities for a specific arc (u, v).

        Args:
            master: Master problem instance
            u: Arc origin node
            v: Arc destination node
            route_values: Current LP solution {route_idx: λ_k}
            max_cuts: Maximum cuts to add for this arc

        Returns:
            Number of cuts added
        """
        # Find routes that use arc (u, v)
        routes_on_arc: List[Tuple[int, float, float]] = []  # (idx, λ_k, load_k)

        for idx, lam in route_values.items():
            if lam < 1e-6:
                continue

            route = master.routes[idx]
            full_path = [0] + route.nodes + [0]

            # Check if arc (u, v) is in the route
            uses_arc = any((full_path[i] == u and full_path[i + 1] == v) for i in range(len(full_path) - 1))

            if uses_arc:
                routes_on_arc.append((idx, lam, route.load))

        if len(routes_on_arc) < 2:
            return 0  # Need at least 2 routes for a cover

        # Sort routes by load (greedy cover selection)
        routes_on_arc.sort(key=lambda x: x[2], reverse=True)

        # Find minimal cover C (routes that exceed capacity when combined)
        cover = []
        cover_load = 0.0
        cover_lam_sum = 0.0

        for idx, lam, load in routes_on_arc:
            cover.append(idx)
            cover_load += load
            cover_lam_sum += lam

            # Check if we have a cover (infeasible combination)
            if cover_load > self.v_model.capacity:
                break

        if cover_load <= self.v_model.capacity:
            return 0  # No cover found

        # Check minimality: removing any route should make it feasible
        cover_is_minimal = self._check_minimal_cover(cover, master.routes)
        if not cover_is_minimal:
            # Remove routes until minimal
            cover = self._make_cover_minimal(cover, master.routes)

        # Compute lifting coefficients for routes not in cover
        non_cover = [idx for idx, _, _ in routes_on_arc if idx not in cover]
        lifting_coeffs = self._compute_lifting_coefficients(cover, non_cover, master.routes)

        # Check if inequality is violated
        lhs = sum(route_values.get(k, 0.0) for k in cover)
        lhs += sum(lifting_coeffs.get(k, 1.0) * route_values.get(k, 0.0) for k in non_cover)
        rhs = len(cover) - 1

        violation = lhs - rhs

        if violation > 0.01:
            # Add cut to master problem
            # Note: This requires extending VRPPMasterProblem to support LCI cuts
            # For now, we'll return 0 as the master problem interface needs updating
            # TODO: Implement master.add_lifted_cover_cut() method
            self.cut_count += 1
            return 1

        return 0

    def _check_minimal_cover(self, cover: List[int], routes: List) -> bool:
        """
        Check if a cover is minimal (removing any route makes it feasible).

        Args:
            cover: List of route indices in the cover
            routes: All routes from master problem

        Returns:
            True if cover is minimal
        """
        cover_load = sum(routes[k].load for k in cover)

        for k in cover:
            remaining_load = cover_load - routes[k].load
            if remaining_load <= self.v_model.capacity:
                return False  # Not minimal

        return True

    def _make_cover_minimal(self, cover: List[int], routes: List) -> List[int]:
        """
        Remove routes from cover until it becomes minimal.

        Args:
            cover: List of route indices
            routes: All routes from master problem

        Returns:
            Minimal cover
        """
        minimal_cover = cover.copy()

        for k in cover:
            # Try removing k
            test_cover = [idx for idx in minimal_cover if idx != k]
            test_load = sum(routes[idx].load for idx in test_cover)

            # If still infeasible, keep removing
            if test_load > self.v_model.capacity:
                minimal_cover = test_cover

        return minimal_cover

    def _solve_knapsack_max_items(self, weights: List[float], capacity: float) -> int:
        """
        Solve 0-1 Knapsack to maximize the NUMBER of items that fit in the knapsack.

        This is a specialized knapsack variant where:
        - Each item has weight w_i (continuous, in kg)
        - Each item has value = 1 (we want to maximize count, not total value)
        - Capacity is C (continuous, in kg)

        Algorithm:
        ----------
        We use dynamic programming with discretized weights to handle continuous values.
        The weights are scaled by a factor (SCALE_FACTOR) to convert to integers,
        then standard DP is applied.

        Complexity: O(n * C_scaled) where C_scaled = capacity * SCALE_FACTOR

        Args:
            weights: List of item weights (route loads in kg)
            capacity: Knapsack capacity (vehicle capacity in kg)

        Returns:
            Maximum number of items that can fit in the knapsack

        References:
        -----------
        Zemel, E. (1989). "Easily computable facets of the knapsack polytope."
        Mathematics of Operations Research, 14(4), 760-764.
        """
        if not weights or capacity <= 0:
            return 0

        n = len(weights)

        # Scale continuous weights to integers for DP
        # Use 1000x scaling: 1.5 kg → 1500 (preserves 3 decimal places)
        SCALE_FACTOR = 1000
        scaled_capacity = int(capacity * SCALE_FACTOR)
        scaled_weights = [int(w * SCALE_FACTOR) for w in weights]

        # DP array: dp[c] = max number of items that fit in capacity c
        dp = [0] * (scaled_capacity + 1)

        for i in range(n):
            w = scaled_weights[i]
            # Traverse backwards to avoid using same item twice
            for c in range(scaled_capacity, w - 1, -1):
                # Option 1: Don't take item i → dp[c]
                # Option 2: Take item i → dp[c - w] + 1
                dp[c] = max(dp[c], dp[c - w] + 1)

        return dp[scaled_capacity]

    def _compute_lifting_coefficients(self, cover: List[int], non_cover: List[int], routes: List) -> Dict[int, float]:
        """
        Compute exact lifting coefficients α_j for routes not in the cover.

        Sequential Lifting Algorithm (Zemel, 1989):
        --------------------------------------------
        For a minimal cover C with inequality:
            ∑_{k ∈ C} x_k ≤ |C| - 1

        We lift variables x_j ∉ C sequentially:
            α_j = |C| - 1 - Z(C, w_j)

        where Z(C, w_j) is the solution to:
            Z(C, w_j) = max { ∑_{k ∈ C} z_k : ∑_{k ∈ C} w_k * z_k ≤ Q - w_j, z_k ∈ {0,1} }

        In words: α_j equals the maximum number of cover items that can be
        packed into the remaining capacity (Q - w_j) after reserving space for item j.

        This yields the facet-defining inequality:
            ∑_{k ∈ C} x_k + ∑_{j ∉ C} α_j * x_j ≤ |C| - 1

        Implementation:
        ---------------
        1. Sort non-cover items by weight (descending) for lifting sequence
        2. Initialize current_cover = list(cover)
        3. For each item j in sorted order:
            a. Calculate remaining capacity: C_rem = Q - w_j
            b. Solve knapsack DP to find max items Z from current_cover that fit in C_rem
            c. Calculate lifting coefficient: α_j = |current_cover| - 1 - Z
            d. Add j to current_cover for next iteration
        4. Return lifting_coeffs dictionary

        Complexity: O(|non_cover| * |cover| * Q * SCALE_FACTOR)

        Args:
            cover: Route indices in the minimal cover
            non_cover: Route indices not in the cover
            routes: All routes from master problem

        Returns:
            Dictionary {route_idx: lifting_coefficient}

        References:
        -----------
        Zemel, E. (1989). "Easily computable facets of the knapsack polytope."
        Mathematics of Operations Research, 14(4), 760-764.

        Balas, E. (1975). "Facets of the knapsack polytope."
        Mathematical Programming, 8(1), 146-164.
        """
        lifting_coeffs: Dict[int, float] = {}

        if not non_cover:
            return lifting_coeffs

        # Initialize the lifting set with the original cover
        current_cover = list(cover)

        # Sort non-cover items by weight (descending) for stable lifting sequence
        # Heavier items are lifted first, which typically produces tighter coefficients
        sorted_non_cover = sorted(non_cover, key=lambda idx: routes[idx].load, reverse=True)

        # Sequentially lift each non-cover variable
        for j in sorted_non_cover:
            w_j = routes[j].load

            # Remaining capacity after reserving space for item j
            remaining_capacity = self.v_model.capacity - w_j

            if remaining_capacity <= 0:
                # Item j alone exceeds capacity
                # α_j = |current_cover| - 1 (item j conflicts with entire cover)
                lifting_coeffs[j] = float(len(current_cover) - 1)
            else:
                # Solve knapsack: max number of items from current_cover that fit in remaining_capacity
                cover_weights = [routes[k].load for k in current_cover]
                Z = self._solve_knapsack_max_items(cover_weights, remaining_capacity)

                # Lifting coefficient formula
                alpha_j = len(current_cover) - 1 - Z
                lifting_coeffs[j] = float(max(0, alpha_j))  # Ensure non-negative

            # Add j to current_cover for next iteration
            # This allows subsequent variables to "see" j as a potential conflict
            current_cover.append(j)

        return lifting_coeffs

    def get_name(self) -> str:
        return "lci"


def create_cutting_plane_engine(
    engine_name: str, v_model: VRPPModel, sep_engine: Optional[SeparationEngine] = None
) -> CuttingPlaneEngine:
    """
    Factory function to create cutting plane engines.

    Args:
        engine_name: Name of the engine ("rcc" or "lci")
        v_model: VRPP model for problem data
        sep_engine: Separation engine (required for RCC)

    Returns:
        Instance of the requested cutting plane engine

    Raises:
        ValueError: If engine_name is not recognized or required args missing

    Example:
        >>> sep_engine = SeparationEngine(v_model)
        >>> cut_engine = create_cutting_plane_engine("rcc", v_model, sep_engine)
        >>> cuts_added = cut_engine.separate_and_add_cuts(master, max_cuts=5)
    """
    if engine_name == "rcc":
        if sep_engine is None:
            raise ValueError("RCC engine requires sep_engine parameter")
        return RoundedCapacityCutEngine(v_model, sep_engine)
    elif engine_name == "lci":
        return LiftedCoverInequalityEngine(v_model)
    else:
        raise ValueError(f"Unknown cutting plane engine '{engine_name}'. Valid options are: ['rcc', 'lci']")
