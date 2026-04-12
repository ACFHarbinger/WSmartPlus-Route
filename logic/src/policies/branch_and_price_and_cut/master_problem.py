"""
Master Problem for Branch-and-Price-and-Cut VRPP.

Implements the Set Partitioning Problem (SPP) formulation where each column
represents a feasible route. The master problem selects routes to maximize
total collected profit across a finite fleet.

Theoretical Basis: Barnhart et al. (1998).
"""

import contextlib
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB

if TYPE_CHECKING:
    from .branching import AnyBranchingConstraint

logger = logging.getLogger(__name__)


class Route:
    """Represents a single route (column) in the master problem."""

    def __init__(
        self,
        nodes: List[int],
        cost: float,
        revenue: float,
        load: float,
        node_coverage: Set[int],
    ) -> None:
        """
        Initialise a route.

        Args:
            nodes: Sequence of customer nodes visited (excluding depot returns).
            cost: Total distance cost of the route.
            revenue: Total revenue from collected waste.
            load: Total waste collected on the route.
            node_coverage: Set of customer node IDs covered by this route.
        """
        self.nodes = nodes
        self.cost = cost
        self.revenue = revenue
        self.load = load
        self.profit = revenue - cost
        self.node_coverage = node_coverage
        self.reduced_cost: Optional[float] = None

    def __repr__(self) -> str:
        return f"Route(nodes={self.nodes}, profit={self.profit:.2f})"


class GlobalCutPool:
    """
    Centralized repository for globally valid inequalities across B&B nodes.

    Philosophy:
    In BPC, separation is expensive. By pooling valid inequalities (RCC, SRI, SEC 2.1)
    globally, we ensure that a cut discovered in one branch tightens the LP bound
    in sibling and child branches immediately, avoiding redundant separation and
    reducing the total number of B&B nodes explored.

    RCC storage note:
        RCC cuts are stored as (node_set, rhs) pairs so that the original RHS
        (= 2*⌈demand(S)/Q⌉, computed at discovery) is faithfully replayed when
        the cut is re-injected at descendant nodes. Storing only the node set
        and hard-coding rhs=1.0 would produce trivially weak cuts.

        Assumption: Customer demands and vehicle capacities are static throughout
        the B&B tree. If these were dynamic (e.g., stochastic demands handled at
        internal nodes), the RHS of purely node-set-based cuts could change,
        invalidating the global mathematical integrity of this archive.
    """

    def __init__(self) -> None:
        """Initialise empty global cut registries."""
        # RCC: maps node_set -> original rhs (2*ceil(demand/Q))
        self.rcc_cuts: Dict[FrozenSet[int], float] = {}
        self.sri_cuts: Set[FrozenSet[int]] = set()
        self.active_sri_vectors: Dict[FrozenSet[int], Dict[str, float]] = {}
        self.sec_cuts: Set[FrozenSet[int]] = set()  # Form 2.1 (Global)
        self.edge_clique_cuts: Set[Tuple[int, int]] = set()
        # LCI: maps node_set -> (rhs, route_coefficients, node_alphas)
        # node_alphas: per-node lifting coefficients for pricing (Barnhart et al. 2000 §4.2)
        self.lci_cuts: Dict[FrozenSet[int], Tuple[float, Dict[int, float], Dict[int, float]]] = {}
        # Optional source arc (i, j) for arc-saturation LCI (SaturatedArcLCIEngine).
        # When set, the pricing dual fires ONLY when the DP traverses that specific arc,
        # not on any visit to a node in the cover set.  None for node/capacity LCI.
        self.lci_arcs: Dict[FrozenSet[int], Optional[Tuple[int, int]]] = {}

    def add_cut(self, cut_type: str, data: Any) -> None:
        """
        Archive a globally valid cut in the pool.

        Global Validity Contract:
            Only archive cuts here that are valid at EVERY node in the B&B tree
            (e.g., RCC, SRI, SEC 2.1). Node-local cuts (branching-dependent SECs,
            lifted cover cuts dependent on local bounds) MUST NOT be added to
            this pool or they will lead to incorrect pruning and loss of optimality.

        Args:
            cut_type: The type of cut ("rcc", "sri", "sec_2.1", "edge_clique", "lci").
            data: Cut-specific data.
                  For "rcc": (FrozenSet[int], rhs: float)
                  For "lci": (FrozenSet[int], rhs: float, route_coefficients: Dict[int, float],
                              node_alphas: Dict[int, float])  — 4-tuple (node_alphas optional)
                  For others: FrozenSet[int] or Tuple[int, int].
        """

        if cut_type == "rcc":
            node_set, rhs = data
            # Only archive if better (tighter) than any existing cut on this set.
            existing = self.rcc_cuts.get(node_set, 0.0)
            if rhs > existing:
                self.rcc_cuts[node_set] = rhs
        elif cut_type == "sri":
            node_set, coeff_vec = data
            self.sri_cuts.add(node_set)
            self.active_sri_vectors[node_set] = coeff_vec
        elif cut_type == "sec_2.1":
            self.sec_cuts.add(data)
        elif cut_type == "edge_clique":
            self.edge_clique_cuts.add(data)
        elif cut_type == "lci":
            # Accept 3-tuple, 4-tuple (+ node_alphas), or 5-tuple (+ arc) variants.
            # 5-tuple: (node_set, rhs, coefficients, node_alphas, arc)
            # 4-tuple: (node_set, rhs, coefficients, node_alphas)
            # 3-tuple: (node_set, rhs, coefficients)
            if len(data) == 5:
                node_set, rhs, coefficients, node_alphas, arc = data
            elif len(data) == 4:
                node_set, rhs, coefficients, node_alphas = data
                arc = None
            else:
                node_set, rhs, coefficients = data
                node_alphas = {}
                arc = None
            self.lci_cuts[node_set] = (rhs, coefficients, node_alphas)
            self.lci_arcs[node_set] = arc

    def apply_to_master(self, master: "VRPPMasterProblem") -> int:
        """
        Inject all pooled global cuts into a fresh Master Problem instance.
        Typically called when entering a new B&B node to tighten the root relaxation.

        Args:
            master: The VRPPMasterProblem instance to receive the cuts.

        Returns:
            Number of cuts successfully applied.
        """
        added = 0
        # RCC: replay with the stored (correct) RHS, not a hard-coded value.
        for node_set, rhs in self.rcc_cuts.items():
            if master.add_capacity_cut(list(node_set), rhs=rhs, _skip_pool=True):
                added += 1
        for nodes in self.sri_cuts:
            if master.add_subset_row_cut(nodes):
                added += 1
        for nodes in self.sec_cuts:
            # Form 2.1 is always global
            if master.add_sec_cut(nodes, rhs=1.0, facet_form="2.1"):
                added += 1
        # Re-inject Edge Clique cuts from the global pool.
        for edge_tuple in self.edge_clique_cuts:
            if master.add_edge_clique_cut(edge_tuple[0], edge_tuple[1]):
                added += 1
        # Re-inject LCI cuts — recompute route coefficients from node_alphas using the
        # current master's route list.  The stale coefficients stored at discovery time
        # reference route indices from the discovery B&B node; at a descendant node the
        # master may have a different route pool, so those indices are meaningless.
        for node_set, lci_data in self.lci_cuts.items():
            if len(lci_data) == 3:
                rhs, _stale_coefficients, node_alphas = lci_data
            else:
                rhs, _stale_coefficients = lci_data  # type: ignore[misc]
                node_alphas = {}
            # Retrieve the source arc stored at discovery time (None for node-capacity LCI).
            lci_arc: Optional[Tuple[int, int]] = self.lci_arcs.get(node_set)
            new_coefficients: Dict[int, float] = {}
            for idx, route in enumerate(master.routes):
                alpha_k = sum(node_alphas.get(n, 1.0 if n in node_set else 0.0) for n in route.node_coverage if n != 0)
                if alpha_k > 1e-6:
                    new_coefficients[idx] = alpha_k
            if master.add_lci_cut(list(node_set), rhs, new_coefficients, node_alphas=node_alphas, arc=lci_arc):
                added += 1
        return added


class VRPPMasterProblem:
    """
    Master Problem (MP) for the Branch-and-Price-and-Cut algorithm.

    Formulates and solves the Set Partitioning Problem (SPP) relaxation of the VRPP.
    The Master Problem is the core coordinator for the column generation loop,
    managing the interaction between the linear relaxation (Gurobi), the
    cutting plane engines, and the branching constraints.

    Mathematical Formulation:
    -------------------------
    Maximize ∑_{k ∈ Ω} p_k * λ_k
    Subject to:
        ∑_{k ∈ Ω} a_{ik} * λ_k = 1           ∀ i ∈ Mandatory Customers (Set Partitioning)
        ∑_{k ∈ Ω} a_{jk} * λ_k ≤ 1           ∀ j ∈ Optional Customers (Set Packing)
        ∑_{k ∈ Ω} λ_k <= K                   (Vehicle Fleet Constraint / Knapsack)
        ∑_{k ∈ Ω} γ_{Sk} * λ_k ≤ RHS_S       (Cutting Planes: RCC, SRI, Fleet Cover)
        λ_k ∈ {0, 1}

    Theoretical Exactness & Farkas Pricing:
    ---------------------------------------
    To maintain theoretical exactness and support Ryan-Foster branching, this
    implementation utilizes strict Set Partitioning for all mandatory nodes.
    In the event of LP infeasibility (common during the early stages of
    branching), the RMP utilizes a 2-Phase approach. Instead of using
    arbitrary Big-M artificial variables which can cause numerical instability,
    we extract the `FarkasDual` ray from Gurobi. This dual ray points in the
    direction of infeasibility, allowing the pricing subproblem to generate
    columns that specifically resolve the unmet coverage requirements.

    Dual Smoothing & Stabilization:
    -------------------------------
    To mitigate the "tailing-off" effect and dual degeneracy in column
    generation, this MP supports Exponential Dual Smoothing (EDS) as
    proposed by Wentges (1997). This stabilizes the dual values π passed
    to the subproblem, accelerating convergence by preventing the
    dual signal from oscillating between extreme basin points.

    Reference:
        - Barnhart, C., et al. (1998, 2000).
        - Wentges, T. (1997). "Weighted Dantzig-Wolfe decomposition."
    """

    def __init__(
        self,
        n_nodes: int,
        mandatory_nodes: Set[int],
        cost_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue_per_kg: float,
        cost_per_km: float,
        vehicle_limit: Optional[int] = None,
        global_cut_pool: Optional[GlobalCutPool] = None,
    ) -> None:
        """
        Initialise the Master Problem for VRPP.

        Framework (Barnhart, Hane, and Vance 2000):
        This class maintains the Restricted Master Problem (RMP) using the
        Barnhart-style sequencing of CG followed by separation.

        Set Partitioning & Branching:
        To maintain Ryan-Foster (1981) compatibility, we use strict equality (== 1)
        for mandatory nodes. Artificial variables (Big-M) are omitted in favor of
        Farkas dual extraction (Phase I), ensuring that primary infeasibility guides
        the search for feasible basis columns.

        Dual Stabilization:
        Includes support for Exponential Dual Smoothing (Wentges, 1997) to
        stabilize the price signal and mitigate the "heading-in" effect.

        Args:
            n_nodes: Number of customer nodes (excluding depot).
            mandatory_nodes: Set of node indices that must be visited.
            cost_matrix: Distance matrix where [0] is the depot.
            wastes: Waste volume (kg) per node.
            capacity: Vehicle payload capacity (kg).
            revenue_per_kg: Revenue coefficient (R).
            cost_per_km: Operating cost coefficient (C).
            vehicle_limit: Maximum fleet size (K).
            global_cut_pool: Central repository for cross-node valid inequalities.
        """
        self.n_nodes = n_nodes
        self.mandatory_nodes = mandatory_nodes
        self.optional_nodes = set(range(1, n_nodes + 1)) - mandatory_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.vehicle_limit = vehicle_limit
        self.global_cut_pool = global_cut_pool or GlobalCutPool()

        # BIG_M Calculation:
        # Strictly exceeds the max possible profit of a single route.
        # Bounded by capacity/min_demand to prevent numerical instability.

        # BIG_M must strictly exceed the profit of the best possible single route.
        # We bound the number of nodes a route can visit by capacity: at most
        # capacity / min_demand nodes fit in one route. This is tighter than n_nodes
        # and avoids inflating the LP penalty coefficient unnecessarily.
        max_single_node_revenue = max(self.wastes.values(), default=0.0) * self.R
        min_demand = max(1.0, min((w for w in self.wastes.values() if w > 0), default=1.0))
        max_nodes_per_route = max(1, int(self.capacity / min_demand))
        max_route_profit = max_single_node_revenue * min(max_nodes_per_route, self.n_nodes)
        self.BIG_M = max(1000.0, 10.0 * max_route_profit)

        self.routes: List[Route] = []

        # Gurobi model
        self.model: Optional[gp.Model] = None
        self.lambda_vars: List[gp.Var] = []

        # artificial_vars removed: Gurobi INFEASIBLE status now signals Phase I.

        self.dual_node_coverage: Dict[int, float] = {}
        self.dual_vehicle_limit: float = 0.0

        # ---- Refactoring: Global Python Column Pool -----------------------
        # Stores columns (routes) that have been removed from the active
        # Gurobi RMP to prevent bloat. When duals change, we check this pool
        # before running the expensive RCSPP pricer.
        self.global_column_pool: List[Route] = []

        # ---- Refactoring: 2-Phase Method ----------------------------------
        # Phase 1: Minimize artificial variables (feasibility)
        # Phase 2: Maximize route profits (optimality)
        self.phase: int = 2  # Default to Phase 2 for backward compatibility

        # Map from FrozenSet[int] (node set S) to Gurobi constraint.
        self.active_src_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_sec_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_sec_cuts_local: Dict[FrozenSet[int], gp.Constr] = {}
        self.active_capacity_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        # LCI constraints (>= K) are kept in a dedicated registry so that their
        # dual extraction (positive for <= in MAX LP) is not confused with RCC
        # duals (negative for >= in MAX LP).
        self.active_lci_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        # Per-node lifting coefficients α_i for each active LCI.
        # Required by the pricing subproblem: c'_lm^k = c_lm^k + π_lm + α_lm^k · γ_lm
        # (Barnhart, Hane, Vance 2000 §4.2).
        self.active_lci_node_alphas: Dict[FrozenSet[int], Dict[int, float]] = {}
        # Source arc (i, j) for arc-saturation LCI cuts.  None for node-capacity LCI.
        # Passed to the pricing DP so it gates the dual on the specific arc traversal.
        self.active_lci_arcs: Dict[FrozenSet[int], Optional[Tuple[int, int]]] = {}

        # Dual mappings
        self.dual_src_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_sec_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_sec_cuts_local: Dict[FrozenSet[int], float] = {}
        self.dual_capacity_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_lci_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_sri_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_edge_clique_cuts: Dict[Tuple[int, int], float] = {}

        # SRI
        self.active_sri_cuts: Dict[FrozenSet[int], gp.Constr] = {}

        # Edge Clique cuts
        # Value is (constraint, {route_idx: lifting_coeff})
        # Key: (min_u, max_v, cover_hash)
        self.active_edge_clique_cuts: Dict[Tuple[int, int], Tuple[gp.Constr, Dict[int, float]]] = {}

        # ---- Task 5: Dual Stabilization (Exponential Smoothing) -----------
        # Reference: Guyenne et al. (1994), Wentges (1997)
        # alpha=0.5 provides a balanced damping of dual oscillations.
        self.dual_smoothing_alpha: float = 0.5
        self.prev_dual_node_coverage: Dict[int, float] = {}
        self.prev_dual_vehicle_limit: float = 0.0
        self.prev_dual_capacity_cuts: Dict[FrozenSet[int], float] = {}
        self.prev_dual_sri_cuts: Dict[FrozenSet[int], float] = {}

        # Farkas duals stored during infeasibility detection
        self.farkas_duals: Dict[str, Dict[Any, float]] = {}

        # Column management
        self.column_deletion_enabled: bool = True
        # Branching awareness (Fix Task 3)
        self.strict_set_partitioning: bool = True

        # Fix 3: Enable dual smoothing flag
        self.enable_dual_smoothing: bool = False  # Off by default for exact operation.

    def add_symmetry_breaking_constraints(self) -> int:
        """
        Task 12 (SOTA): Symmetry breaking for identical vehicles.
        Enforces a lexicographical ordering on the routes selected in the solution.
        This prevents the solver from exploring all K! permutations of the same routes.

        Mathematical Formulation:
            bitmask(route_k) >= bitmask(route_{k+1})

        Wait: In the Set Partitioning formulation, λ_k are binary selection variables
        for specific routes. Since we have a vehicle fleet constraint Σ λ_k <= K,
        symmetry only exists if we use a formulation with vehicle-indexed variables λ_{kv}.

        However, in the standard column generation framework, we don't index columns by vehicle.
        Symmetry is *implicitly* handled by the SPP formulation itself (multi-set packing).

        BUT, if we use branching on vehicle-route assignment or hierarchical vehicles,
        then symmetry becomes an issue.

        Correction: For pure VRPP SPP, symmetry breaking is not needed at the Master level.
        It is needed at the *Pricing* level if we have multiple identical vehicles
        with different characteristics (not our case).

        So, Task 12 for VRPP actually refers to **Pruning Equivalent Routes** in the pool.
        Implemented as `deduplicate_column_pool`.
        """
        return 0

    def purge_useless_columns(self, tolerance: float = -0.1) -> int:
        """
        Remove non-basic columns with significantly negative reduced cost.

        In a maximization LP, non-basic columns have RC <= 0. If a column is
        not in the basis and its reduced cost is highly negative, it is
        unlikely to enter the basis soon. Removing such columns prevents
        basis matrix bloat and keeps the RMP solver fast.

        Args:
            tolerance: Reduced cost threshold. Columns with RC < tolerance
                that are non-basic will be removed.

        Returns:
            Number of columns purged from the Gurobi model.
        """
        if not self.column_deletion_enabled or self.model is None or not self.lambda_vars:
            return 0

        # We must solve the model first to have valid Basis/RC attributes.
        # This check happens inside the try-block below.

        to_remove: List[int] = []
        for i, var in enumerate(self.lambda_vars):
            try:
                # var.VBasis: 0=Basic, 1=Lower, 2=Upper, 3=Superbasic
                # We only remove non-basic variables (VBasis != 0)
                if var.VBasis != 0 and tolerance > var.RC:
                    to_remove.append(i)
            except (gp.GurobiError, AttributeError):
                # RC or VBasis might not be available if node was pruned or infeasible
                continue

        if not to_remove:
            return 0

        # Remove columns from Gurobi and move to global pool.
        # We must delete in reverse index order to maintain valid indexing.
        for i in sorted(to_remove, reverse=True):
            # Move to Python Pool before Gurobi deletion
            self.global_column_pool.append(self.routes[i])

            self.model.remove(self.lambda_vars[i])
            self.lambda_vars.pop(i)
            self.routes.pop(i)

        self.model.update()
        logger.info(
            f"[Purge] Removed {len(to_remove)} non-basic columns (RC < {tolerance}). "
            f"Active RMP size: {len(self.routes)}"
        )
        return len(to_remove)

    def sift_global_column_pool(
        self,
        node_duals: Dict[int, float],
        rcc_duals: Dict[FrozenSet[int], float],
        sri_duals: Dict[FrozenSet[int], float],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_duals: Optional[Dict[FrozenSet[int], float]] = None,
        lci_node_alphas: Optional[Dict[FrozenSet[int], Dict[int, float]]] = None,
        branching_constraints: Optional[List["AnyBranchingConstraint"]] = None,
        rc_tolerance: float = 1e-5,
    ) -> int:
        """
        Scan the global pool for profitable columns under current duals and branching.

        Args:
            node_duals: Base duals from coverage and vehicle limits.
            rcc_duals: Current Root-Capacity Cut duals.
            sri_duals: Current Subset-Row Inequality duals.
            edge_clique_duals: Current Edge Clique cut duals.
            lci_duals: Current Lifted Cover Inequality duals (γ per cover set S).
            lci_node_alphas: Per-node lifting coefficients for LCI pricing adjustment.
            branching_constraints: List of active B&B branching constraints.
            rc_tolerance: Numerical threshold to prevent injection of mathematically
                         insignificant columns (epsilon deadlock).

        Returns:
            Number of routes re-activated and added to the RMP.
        """
        if not self.global_column_pool:
            return 0

        # Bundle duals for calculate_reduced_cost
        dual_values: Dict[str, Any] = {
            "node_duals": node_duals,
            "rcc_duals": rcc_duals,
            "sri_duals": sri_duals,
            "edge_clique_duals": edge_clique_duals,
            "lci_duals": lci_duals or {},
            "lci_node_alphas": lci_node_alphas or {},
        }

        # We use a set of signatures to prevent adding duplicates if they already exist in RMP
        # (Though column generation usually handles this via dominance).
        active_sigs = {tuple(r.nodes) for r in self.routes}

        # Task 6 (SOTA): Elite Sifting.
        # Instead of adding all profitable columns, we sort by current reduced cost
        # and only inject the top 'max_to_add' (N=30) movers. This prevents RMP bloat.
        max_to_add = 30
        candidates: List[Tuple[float, int]] = []

        for i, route in enumerate(self.global_column_pool):
            # Task 1: Enforce physical elementarity.
            if len(set(route.nodes)) != len(route.nodes):
                continue

            # Check for duplicates in active model
            if tuple(route.nodes) in active_sigs:
                continue

            # Task 6: Check feasibility against branching constraints.
            if branching_constraints and not all(bc.is_route_feasible(route) for bc in branching_constraints):
                continue

            # Task 1: Enforce rc_tolerance
            rc = self.calculate_reduced_cost(route, dual_values)
            if rc > rc_tolerance:
                candidates.append((rc, i))

        if not candidates:
            return 0

        # SORT BY QUALITY (Elite Sifting)
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_elite = candidates[:max_to_add]

        # Extract only indices for removal (reverse to preserve indexing)
        top_indices = sorted([idx for _, idx in top_elite], reverse=True)

        added = 0
        for idx in top_indices:
            route = self.global_column_pool.pop(idx)
            self.add_route(route)
            added += 1

        if added > 0 and self.model is not None:
            self.model.update()
            logger.info(f"[Sifting] Re-added {added} elite columns from pool (cap={max_to_add}).")
        return added

    def calculate_reduced_cost(self, route: Route, dual_values: Dict[str, Any]) -> float:
        """
        Helper to calculate reduced cost of a route using current duals.
        Correctly accounts for node coverage, fleet limit, and all active cuts.
        """
        node_duals = dual_values.get("node_duals", {})

        # Reduced cost base: Profit - Σ duals
        # For Set Packing (>= 1 or == 1), duals π_i are represented such that
        # rc = p_k - Σ a_ik * π_i.
        rc = route.revenue - route.cost

        # 1. Node coverage duals (Partitioning/Packing)
        for node in route.node_coverage:
            rc -= node_duals.get(node, 0.0)

        # 2. Vehicle limit dual
        # If dual_values contains the key, it's already extracted from get_reduced_cost_coefficients
        # which bundles it into node_duals["vehicle_limit"].
        rc -= node_duals.get("vehicle_limit", 0.0)

        # 3. Capacity Cut (RCC) duals
        rcc_duals = dual_values.get("rcc_duals", {})
        for node_set, dual in rcc_duals.items():
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                rc -= float(crossings) * dual

        # 4. Subset-Row Inequality (SRI) duals
        sri_duals = dual_values.get("sri_duals", {})
        route_nodes = set(route.nodes)
        for subset, dual in sri_duals.items():
            count = len(subset.intersection(route_nodes))
            coeff = count // 2
            if coeff > 0:
                rc -= float(coeff) * dual

        # 5. Edge Clique duals
        edge_clique_duals = dual_values.get("edge_clique_duals", {})
        nodes = [0] + route.nodes + [0]
        for i in range(len(nodes) - 1):
            edge = (min(nodes[i], nodes[i + 1]), max(nodes[i], nodes[i + 1]))
            if edge in edge_clique_duals:
                rc -= edge_clique_duals[edge]

        # 6. Lifted Cover Inequality (LCI) duals — Barnhart et al. (2000) §4.2
        # For each active LCI with cover set S and dual γ_S, the reduced cost
        # contribution is:  rc -= Σ_{i ∈ route ∩ S} α_i · γ_S
        # where α_i is the node-level lifting coefficient.
        lci_duals = dual_values.get("lci_duals", {})
        lci_node_alphas = dual_values.get("lci_node_alphas", {})
        if lci_duals:
            route_nodes_set = route.node_coverage
            for cover_set, dual in lci_duals.items():
                if dual < 1e-9:
                    continue
                node_alpha = lci_node_alphas.get(cover_set, {})
                for node in route_nodes_set:
                    alpha = node_alpha.get(node, 1.0 if node in cover_set else 0.0)
                    if alpha > 1e-9:
                        rc -= alpha * dual

        return rc

    def get_dual_values(self) -> Dict[str, Any]:
        """
        Bundle all active dual variables into a structured dictionary.
        Alias for get_reduced_cost_coefficients() used for pool sifting.
        """
        return self.get_reduced_cost_coefficients()

    def set_phase(self, phase: int) -> None:
        """
        Switch between Phase 1 (Feasibility) and Phase 2 (Optimality).

        Phase 1: All route objectives are set to 0.0 so the LP has no inherent
            bias.  Gurobi will find the RMP infeasible (no Big-M artificials),
            and the Farkas dual ray guides pricing toward columns that restore
            coverage of mandatory nodes.
        Phase 2: Route objectives are restored to their profit values so the LP
            maximises total profit (standard column generation).

        Args:
            phase: 1 or 2.
        """
        if self.model is None:
            return

        self.phase = phase
        if phase == 1:
            logger.info("Master Problem: Switching to Phase I (Feasibility).")
            # Phase I: all route variables have zero obj. coefficient.
            # Farkas dual extraction will guide pricing to find a feasible basis.
            for var in self.lambda_vars:
                var.Obj = 0.0
            self.model.ModelSense = GRB.MAXIMIZE
        else:
            logger.info("Master Problem: Switching to Phase II (Optimality).")
            # Phase II: standard profit maximisation.
            for idx, route in enumerate(self.routes):
                self.lambda_vars[idx].Obj = route.profit
            self.model.ModelSense = GRB.MAXIMIZE

        self.model.update()

    # ------------------------------------------------------------------
    # Column management
    # ------------------------------------------------------------------

    def add_route_as_column(self, route: Route) -> None:
        """
        Add a route as a column to the live Gurobi model.

        Alias for add_route — exists so that column-generation callers can
        distinguish between buffered pre-build additions (add_route before
        build_model) and live column insertions during CG (add_route_as_column
        after build_model).  Both delegate to _add_column_to_model when the
        model is already built.

        Args:
            route: Route to insert as a new λ variable.
        """
        self.add_route(route)

    def add_route(self, route: Route) -> None:
        """
        Add a route (column) to the master problem.

        If the Gurobi model has already been built, the column is inserted
        into the live model immediately.  Otherwise it is buffered in
        self.routes and will be included when build_model() is called.

        Args:
            route: Route to add.
        """
        self.routes.append(route)
        if self.model is not None:
            self._add_column_to_model(route)

    def _add_column_to_model(self, route: Route) -> None:  # noqa: C901
        """
        Insert a new column (route variable) into the live Gurobi model.

        Args:
            route: Route whose λ variable is being added.
        """
        if self.model is None:
            return

        var = self.model.addVar(
            obj=route.profit,
            vtype=GRB.BINARY,
            name=f"route_{len(self.lambda_vars)}",
        )

        # Wire the variable into every coverage constraint it participates in.
        for node in route.node_coverage:
            constr = self.model.getConstrByName(f"coverage_{node}")
            if constr is not None:
                self.model.chgCoeff(constr, var, 1.0)

        # Wire into the vehicle limit constraint if present.
        if self.vehicle_limit is not None:
            constr = self.model.getConstrByName("vehicle_limit")
            if constr is not None:
                self.model.chgCoeff(constr, var, 1.0)

        self.lambda_vars.append(var)
        self._wire_route_into_active_cuts(route, var)
        self.model.update()

    def _wire_route_into_active_cuts(self, route: Route, var: Any) -> None:
        """
        Wire a route variable into all currently active cutting planes.

        This ensures that columns added after some cuts have been discovered
        (standard Column Generation + Cutting Planes sequence) correctly
        participate in those cuts, maintaining LP relaxation validity.
        """
        if self.model is None:
            return

        # 1. Wire into active capacity cuts (RCC)
        for node_set, constr in self.active_capacity_cuts.items():
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                self.model.chgCoeff(constr, var, float(crossings))

        # 2. Wire into active SEC cuts (Form 2.1, 2.2, 2.3)
        # Check both global and local-node SEC pools.
        for node_set, constr in self.active_sec_cuts.items():
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                self.model.chgCoeff(constr, var, float(crossings))

        for node_set, constr in self.active_sec_cuts_local.items():
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                self.model.chgCoeff(constr, var, float(crossings))

        # 3. Wire into active Subset-Row Inequalities (SRI)
        for subset, constr in self.active_sri_cuts.items():
            count = sum(1 for n in subset if n in route.node_coverage)
            coeff_sri = count // 2
            if coeff_sri > 0:
                self.model.chgCoeff(constr, var, float(coeff_sri))

        # 4. Wire into active Edge Clique cuts
        route_idx = len(self.lambda_vars) - 1
        for (u, v), (constr, stored_coeffs) in self.active_edge_clique_cuts.items():
            nodes = [0] + route.nodes + [0]
            contains_edge = any(tuple(sorted((nodes[i], nodes[i + 1]))) == (u, v) for i in range(len(nodes) - 1))
            if contains_edge:
                # Per Barnhart et al. (2000), new routes not present in the cut discovery
                # default to coefficient 1.0 for edge-cover based cliques.
                coeff_clique = stored_coeffs.get(route_idx, 1.0)
                self.model.chgCoeff(constr, var, float(coeff_clique))

        # 5. Wire into active LCI cuts — recompute per-route coefficient from stored node alphas.
        # Routes added after an LCI cut was created would have coefficient 0 by default,
        # making the LCI constraint mathematically under-constrained.  We fix this by
        # computing α_k = Σ_{i ∈ route} α_i using the archived per-node lifting coefficients.
        for node_set, constr in self.active_lci_cuts.items():
            node_alphas_for_set = self.active_lci_node_alphas.get(node_set, {})
            alpha_k = sum(
                node_alphas_for_set.get(n, 1.0 if n in node_set else 0.0) for n in route.node_coverage if n != 0
            )
            if alpha_k > 1e-6:
                self.model.chgCoeff(constr, var, alpha_k)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self, initial_routes: Optional[List[Route]] = None) -> None:
        """
        Build the Gurobi model for the set-covering master problem.

        Implements the Restricted Master Problem (RMP) formulation:

        1. Mandatory-node coverage constraints use == 1 (Set Partitioning) to
           ensure that Ryan-Foster branching logic remains mathematically valid.
        2. Artificial variables (Big-M) are intentionally omitted.  When the
           column pool cannot cover a mandatory node, Gurobi returns INFEASIBLE
           and we extract the Farkas dual ray to guide Phase I pricing toward
           feasibility (Barnhart et al. 2000, §3).  Big-M variables would
           suppress infeasibility and deadlock the Phase I / Farkas loop.
        3. Optional-node constraints are packing inequalities (<= 1).

        Args:
            initial_routes: If provided, replaces self.routes with this list
                before building the model.
        """
        self.model = gp.Model("VRPP_Master")
        self.model.Params.OutputFlag = 0
        # Use the Dual Simplex algorithm to solve the LP relaxation.
        # This produces a valid basis for warm-starting across B&B nodes.
        # Reference: Barnhart et al. (1998).
        self.model.Params.Method = 1
        self.model.Params.InfUnbdInfo = 1
        # Task 8: Prevent the Gurobi DualReductions Trap.
        # Disabling DualReductions forces Gurobi to use the dual simplex method to
        # prove infeasibility, which ensures that a valid Farkas dual ray is always
        # mathematically available for extraction during Phase I pricing.
        self.model.Params.DualReductions = 0
        self.lambda_vars = []
        if initial_routes is not None:
            self.routes = initial_routes

        # ---- Route (λ) variables ----------------------------------------
        for idx, route in enumerate(self.routes):
            var = self.model.addVar(
                obj=route.profit,
                vtype=GRB.BINARY,
                name=f"route_{idx}",
            )
            self.lambda_vars.append(var)

        # ---- Mandatory-node coverage constraints (Set Partitioning: == 1) ----
        # Note: artificial variables (Big-M slack) are intentionally OMITTED.
        # Without them, Gurobi can naturally return GRB.INFEASIBLE when the
        # current column pool cannot cover a mandatory node — this is the correct
        # trigger for Phase I Farkas pricing. Adding artificials permanently
        # suppresses infeasibility and deadlocks the Phase I / Farkas loop.
        for node in self.mandatory_nodes:
            lhs = gp.LinExpr()
            for idx, route in enumerate(self.routes):
                if node in route.node_coverage:
                    lhs += self.lambda_vars[idx]

            if self.strict_set_partitioning:
                self.model.addConstr(lhs == 1.0, name=f"coverage_{node}")
            else:
                self.model.addConstr(lhs >= 1.0, name=f"coverage_{node}")

        # ---- Optional-node packing constraints (<= 1) --------------------
        # Optional nodes may be visited at most once across all selected routes.
        # No artificial variable is needed: being unvisited (lhs = 0) is valid.
        for node in self.optional_nodes:
            lhs = gp.LinExpr()
            for idx, route in enumerate(self.routes):
                if node in route.node_coverage:
                    lhs += self.lambda_vars[idx]

            self.model.addConstr(
                lhs <= 1,
                name=f"coverage_{node}",
            )

        # ---- Fleet-size constraint (optional) ----------------------------
        if self.vehicle_limit is not None:
            self.model.addConstr(
                gp.quicksum(self.lambda_vars) <= self.vehicle_limit,
                name="vehicle_limit",
            )

        self.model.ModelSense = GRB.MAXIMIZE
        self.model.update()

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def solve_lp_relaxation(self) -> Tuple[Optional[float], Dict[int, float]]:  # noqa: C901
        """
        Solve the LP relaxation of the RMP and extract dual values.

        The integrality constraints on all λ variables are temporarily relaxed
        to continuous [0, 1]. The model is solved using the Dual Simplex method
        (Reference: Barnhart et al. 1998) to ensure efficient warm-starting.

        Dual Extraction & Interpretation:
        - Mandatory Nodes (== 1): π_i is the shadow price.
        - Optional Nodes (<= 1): π_j >= 0.
        - Vehicle Limit (<= K): π_K >= 0 (fleet opportunity cost).
        - Cuts (Σ aλ <= RHS): Dual weights used to penalize "violated" routes in pricing.

        Infeasibility & Farkas:
        If the current column pool cannot satisfy mandatory coverage, Gurobi returns
        GRB.INFEASIBLE. We then extract the Farkas Dual Ray (FarkasDual) which
        mathematically identifies the direction of infeasibility, guiding the
        pricing subproblem to find feasible columns (Phase I).

        Returns:
            Tuple of (LP objective value, {route_index: λ_k value}).
            If infeasible, returns (-inf, {}) and populates self.farkas_duals.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Relax integrality on route variables (λ → continuous).
        for var in self.lambda_vars:
            var.VType = GRB.CONTINUOUS

        # Fix 14: Basis reuse. In BPC, RMP basis stability is critical for
        # performance. Gurobi naturally warm-starts from the previous basis if the
        # model is only updated. By ensuring we only call update() and not reset(),
        # we maximize the speed of solving sibling/child nodes.
        self.model.update()
        if self.model.NumVars == 0:
            return 0.0, {}
        self.model.optimize()
        status = self.model.Status

        if status == GRB.INFEASIBLE:
            try:
                # Structure farkas_duals identically to get_reduced_cost_coefficients()
                farkas_node_duals: Dict[Union[int, str], float] = {}
                for node in range(1, self.n_nodes + 1):
                    constr = self.model.getConstrByName(f"coverage_{node}")
                    if constr is not None:
                        farkas_node_duals[node] = constr.FarkasDual

                if self.vehicle_limit is not None:
                    constr = self.model.getConstrByName("vehicle_limit")
                    if constr is not None:
                        farkas_node_duals["vehicle_limit"] = constr.FarkasDual

                # Task 3: Extract Minimum Fleet Farkas Duals
                temp_min_c = self.model.getConstrByName("temp_min_vehicles")
                if temp_min_c is not None:
                    farkas_node_duals["vehicle_limit"] = (
                        farkas_node_duals.get("vehicle_limit", 0.0) + temp_min_c.FarkasDual
                    )

                farkas_rcc_duals: Dict[FrozenSet[int], float] = {}
                for subset, constr in self.active_capacity_cuts.items():
                    farkas_rcc_duals[subset] = constr.FarkasDual

                farkas_sri_duals: Dict[FrozenSet[int], float] = {}
                for subset, constr in self.active_sri_cuts.items():
                    farkas_sri_duals[subset] = constr.FarkasDual

                farkas_clique_duals: Dict[Tuple[int, int], float] = {}
                for edge, (constr, _) in self.active_edge_clique_cuts.items():
                    farkas_clique_duals[edge] = constr.FarkasDual

                self.farkas_duals = {
                    "node_duals": farkas_node_duals,
                    "rcc_duals": farkas_rcc_duals,
                    "sri_duals": farkas_sri_duals,
                    "edge_clique_duals": farkas_clique_duals,
                }
                return -float("inf"), {}
            except AttributeError:
                # Big-M Fallback: If Farkas dual extraction fails (e.g. numeric issues
                # or DualReductions=1 override), artificially force feasibility with
                # highly penalized Big-M slacks to extract exact surrogate Farkas penalties.
                logger.warning("FarkasDual extraction failed. Falling back to Big-M artificial variables.")

                artificial_vars = []
                big_m = -1e6
                # We need to satisfy mandatory node coverage constraints
                for node in self.mandatory_nodes:
                    constr = self.model.getConstrByName(f"coverage_{node}")
                    if constr is not None:
                        # Add variable with high penalty to bridge the infeasibility
                        art = self.model.addVar(obj=big_m, vtype=GRB.CONTINUOUS, name=f"art_cov_{node}")
                        self.model.chgCoeff(constr, art, 1.0)
                        artificial_vars.append(art)

                self.model.update()
                self.model.optimize()

                if self.model.Status == GRB.OPTIMAL:
                    # Extract standard duals directly since artificials bridged the infeasibility.
                    # Because the objective penalty is huge negative, the Pi values will be
                    # huge negative/positive accordingly, acting exactly as scaled Farkas duals.
                    self.farkas_duals = self.get_reduced_cost_coefficients()
                else:
                    self.farkas_duals = {"node_duals": {}, "rcc_duals": {}, "sri_duals": {}, "edge_clique_duals": {}}

                for art in artificial_vars:
                    self.model.remove(art)
                self.model.update()
                return -float("inf"), {}

        if status != GRB.OPTIMAL:
            # Handle other non-optimal statuses (e.g. time limit, numeric issues)
            logger.warning(f"LP solve returned non-optimal status {status}. Treating node as infeasible.")
            return None, {}

        obj_value = self.model.ObjVal
        try:
            route_values = {idx: var.X for idx, var in enumerate(self.lambda_vars)}
        except (gp.GurobiError, AttributeError):
            return 0.0, {}

        # ---- Extract dual values -----------------------------------------
        # For a >= constraint in a MAX LP, Gurobi reports a non-positive π.
        # The pricing subproblem expects non-negative duals (interpreted as the
        # "value" of covering a node), so we store |π|.
        self.dual_node_coverage = {}
        for node in range(1, self.n_nodes + 1):
            self.dual_node_coverage[node] = 0.0
            if self.model is None or self.model.NumVars == 0 or self.model.Status != GRB.OPTIMAL:
                continue

            constr = self.model.getConstrByName(f"coverage_{node}")
            if constr is None:
                continue

            try:
                if node in self.mandatory_nodes:
                    # == 1 constraint (Set Partitioning) in MAX LP.
                    # Gurobi reports Pi = dObjValue/dRHS directly.
                    # The pricing reduced cost is: rc = profit - cost - Σπ_i.
                    # We must use Pi as-is (not negated), so the DP subtracts
                    # the correct shadow price and searches toward positive rc columns.
                    self.dual_node_coverage[node] = constr.Pi
                else:
                    # <= 1 constraint in MAX LP: Pi >= 0, dual value = Pi >= 0.
                    self.dual_node_coverage[node] = max(0.0, constr.Pi)
            except gp.GurobiError:
                continue

        # For the <= vehicle-limit constraint in a MAX LP, the dual is the
        # opportunity cost of using one more vehicle; clamp to non-negative.
        self.dual_vehicle_limit = 0.0
        if (
            self.vehicle_limit is not None
            and self.model is not None
            and self.model.NumVars > 0
            and self.model.Status == GRB.OPTIMAL
        ):
            constr = self.model.getConstrByName("vehicle_limit")
            if constr is not None:
                # Task 2 (Regression Fix): Gurobi returns non-negative dual prices (Pi)
                # for <= constraints in a MAXIMIZATION model.
                with contextlib.suppress(gp.GurobiError):
                    self.dual_vehicle_limit = max(0.0, constr.Pi)

            # Task 3: Extract Minimum Fleet Duals
            temp_min_c = self.model.getConstrByName("temp_min_vehicles")
            if temp_min_c is not None:
                with contextlib.suppress(gp.GurobiError):
                    # For >= constraints in MAX LP, Pi is non-positive.
                    # We add |Pi| (shadow value) to the fleet penalty.
                    self.dual_vehicle_limit += max(0.0, -temp_min_c.Pi)

        # ---- Extract duals for capacity cuts -----------------------------
        self.exact_dual_capacity_cuts = {}
        self.dual_capacity_cuts = {}
        for node_set, constr in self.active_capacity_cuts.items():
            val = 0.0
            if self.model.Status == GRB.OPTIMAL:
                with contextlib.suppress(gp.GurobiError, AttributeError):
                    val = abs(constr.Pi)

            # Store exact mathematically pure dual for bound calculation
            self.exact_dual_capacity_cuts[node_set] = val

            # Apply Wentges heuristic smoothing if enabled
            use_smoothing = self.enable_dual_smoothing and not getattr(self, "_exact_mode_active", False)
            if use_smoothing:
                prev_val = self.prev_dual_capacity_cuts.get(node_set, 0.0)
                val = self.dual_smoothing_alpha * val + (1 - self.dual_smoothing_alpha) * prev_val
                self.prev_dual_capacity_cuts[node_set] = val
            self.dual_capacity_cuts[node_set] = val

        # ---- Extract duals for SRI ---------------------------------------
        # SRI cuts are <= 1 constraints.  In a MAX LP, Gurobi reports Pi >= 0
        # for every <= constraint (relaxing the upper bound can only help the
        # objective).  We therefore use +Pi, not -Pi.
        # Bug history: using max(0, -Pi) always yielded 0 because -Pi <= 0,
        # so SRI cuts never influenced the pricing subproblem.
        self.exact_dual_sri_cuts = {}
        self.dual_sri_cuts = {}
        for subset, constr in self.active_sri_cuts.items():
            val = 0.0
            if self.model.Status == GRB.OPTIMAL:
                with contextlib.suppress(gp.GurobiError, AttributeError):
                    val = abs(constr.Pi)

            self.exact_dual_sri_cuts[subset] = val

            if self.enable_dual_smoothing and not getattr(self, "_exact_mode_active", False):
                prev_val = self.prev_dual_sri_cuts.get(subset, 0.0)
                val = self.dual_smoothing_alpha * val + (1 - self.dual_smoothing_alpha) * prev_val
                self.prev_dual_sri_cuts[subset] = val
            self.dual_sri_cuts[subset] = val

        # Extract Edge Clique cut duals
        # Edge cliques are <= 1 constraints in MAX LP — same sign convention as SRI.
        # Pi >= 0; use +Pi directly.
        self.dual_edge_clique_cuts = {}
        for key, (constr, _) in self.active_edge_clique_cuts.items():
            if constr is not None:
                with contextlib.suppress(gp.GurobiError):
                    self.dual_edge_clique_cuts[key] = max(0.0, constr.Pi)

        # ---- Extract duals for LCI cuts ----------------------------------
        # LCI cuts are <= rhs constraints.  In a MAX LP, Pi >= 0 for <=
        # constraints, so we take Pi directly (no negation).
        self.dual_lci_cuts = {}
        for node_set, constr in self.active_lci_cuts.items():
            if constr is not None:
                with contextlib.suppress(gp.GurobiError):
                    self.dual_lci_cuts[node_set] = max(0.0, constr.Pi)

        # ---- Task 5: Apply Exponential Smoothing -----------------------
        # Strictly separate exact duals before any smoothing is applied in-place
        self.exact_duals = {
            "node_duals": self.dual_node_coverage.copy(),
            "rcc_duals": getattr(self, "exact_dual_capacity_cuts", self.dual_capacity_cuts.copy()),
            "sri_duals": getattr(self, "exact_dual_sri_cuts", self.dual_sri_cuts.copy()),
            "edge_clique_duals": self.dual_edge_clique_cuts.copy(),
            "lci_duals": self.dual_lci_cuts.copy(),
            "vehicle_limit": self.dual_vehicle_limit,
        }

        if self.enable_dual_smoothing and not getattr(self, "_exact_mode_active", False):
            self.smoothed_duals_active = True
            self._apply_dual_smoothing()
        else:
            self.smoothed_duals_active = False

        return obj_value, route_values

    def _apply_dual_smoothing(self) -> None:
        """
        Apply Exponential Dual Smoothing to stabilize CG price signals.
        pi_smoothed = alpha * pi_current + (1 - alpha) * pi_prev

        WARNING: Smoothing modifies self.dual_node_coverage in-place.
        When active, the duals passed to pricing are NOT the true LP duals.
        This means:
        - The Lagrangian upper bound (obj_val + K * max_rc) is not valid.
        - CG convergence (added == 0) does not prove LP optimality.
        Only enable this flag when running in heuristic (non-exact) mode.

        Reference: Wentges (1997), Guyenne et al. (1994).
        """
        alpha = self.dual_smoothing_alpha

        # 1. Node coverage duals
        for node, current_val in self.dual_node_coverage.items():
            prev_val = self.prev_dual_node_coverage.get(node, current_val)
            self.dual_node_coverage[node] = alpha * current_val + (1.0 - alpha) * prev_val
        self.prev_dual_node_coverage = self.dual_node_coverage.copy()

        # 2. Vehicle limit dual
        prev_limit_val = self.prev_dual_vehicle_limit
        self.dual_vehicle_limit = alpha * self.dual_vehicle_limit + (1.0 - alpha) * prev_limit_val
        self.prev_dual_vehicle_limit = self.dual_vehicle_limit

    def solve_ip(self) -> Tuple[float, List[Route]]:
        """
        Solve the integer programme at the current B&B node.

        Re-imposes binary integrality on all λ variables and calls Gurobi's
        MIP solver.  Artificial variables are left continuous (they will be
        forced to zero by the Big-M penalty if any feasible integer solution
        exists).

        Returns:
            Tuple of:
                obj_value       – IP objective value.
                selected_routes – Routes with λ_k = 1 in the optimal solution.

        Raises:
            ValueError: If build_model() has not been called.
            RuntimeError: If the MIP solve fails.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        if self.model.NumVars == 0:
            return 0.0, []

        for var in self.lambda_vars:
            var.VType = GRB.BINARY

        self.model.update()

        try:
            self.model.optimize()

            if self.model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                raise RuntimeError(f"IP solving failed with status {self.model.Status}")

            obj_value = self.model.ObjVal
            selected_routes = [self.routes[idx] for idx, var in enumerate(self.lambda_vars) if var.X > 0.5]
            return obj_value, selected_routes
        finally:
            # Always restore LP-relaxation variable types so subsequent calls to
            # solve_lp_relaxation work correctly even after an IP solve failure.
            for var in self.lambda_vars:
                var.VType = GRB.CONTINUOUS
            self.model.update()

    # ------------------------------------------------------------------
    # Dual / pricing helpers
    # ------------------------------------------------------------------

    def deduplicate_column_pool(self, tol: float = 1e-6) -> int:
        """
        Fix 18: Prune mathematically equivalent routes from the global column pool.
        Uses MD5 content hashes of node sequences for O(1) deduplication.
        """
        import hashlib

        if not self.global_column_pool:
            return 0

        initial_count = len(self.global_column_pool)
        seen_hashes: Dict[str, float] = {}
        unique_pool = []

        for route in self.global_column_pool:
            content = ",".join(map(str, route.nodes))
            h = hashlib.md5(content.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes[h] = route.profit
                unique_pool.append(route)
            else:
                if route.profit > seen_hashes[h] + tol:
                    # Replace existing with higher profit version
                    for idx, r in enumerate(unique_pool):
                        r_content = ",".join(map(str, r.nodes))
                        rh = hashlib.md5(r_content.encode()).hexdigest()
                        if rh == h:
                            unique_pool[idx] = route
                            seen_hashes[h] = route.profit
                            break

        self.global_column_pool = unique_pool
        return initial_count - len(unique_pool)

    def get_reduced_cost_coefficients(self) -> Dict[str, Dict[Union[int, FrozenSet[int], str, Tuple[int, int]], float]]:
        """
        Return the dual-value coefficients used by the pricing subproblem.

        Sign Convention & Maximization:
            The VRPP is formulated as a maximization problem. Under Gurobi's
            standard dual representation, the reduced cost calculation for
            pricing (searching for columns to boost the objective) is:
            rc(r) = profit(r) - Σ (a_{ik} * π_i), where π_i are the duals.
            - For >= constraints (e.g. RCC), Pi is non-positive; we use -Pi.
            - For <= constraints (e.g. SRI, Vehicle Limit), Pi is non-negative; we use +Pi.

        Returns:
            Dictionary mapping node ID (int) → dual value, plus the key
            "vehicle_limit" → vehicle-limit dual if a fleet cap is active.
        """
        assert self.model is not None and self.model.ModelSense == GRB.MAXIMIZE, (
            "Reduced cost extraction assumes a MAXIMIZATION master problem."
        )

        duals: Dict[Union[int, str, frozenset[int], Tuple[int, int]], float] = {
            k: v for k, v in self.dual_node_coverage.items()
        }
        if self.vehicle_limit is not None:
            duals["vehicle_limit"] = self.dual_vehicle_limit

        # Sum Edge Clique duals by edge for the subproblem
        clique_duals_collapsed: Dict[Union[int, str, FrozenSet[int], Tuple[int, int]], float] = {}
        for (u, v), dual in self.dual_edge_clique_cuts.items():
            clique_duals_collapsed[(u, v)] = clique_duals_collapsed.get((u, v), 0.0) + dual

        # Use a type-safe casting by converting to the target dictionary type explicitly
        result: Dict[str, Any] = {
            "node_duals": {k: v for k, v in duals.items()},
            "rcc_duals": {k: v for k, v in self.dual_capacity_cuts.items()},
            "sri_duals": {k: v for k, v in self.dual_sri_cuts.items()},
            "edge_clique_duals": {k: v for k, v in clique_duals_collapsed.items()},
            # LCI duals (γ) and per-node lifting coefficients (α_i) for pricing.
            # These allow the RCSPP to deduct α_i · γ when visiting node i in cover S,
            # implementing Barnhart et al. (2000) §4.2: c'_lm^k = c_lm^k + π_lm + α_lm^k · γ_lm
            "lci_duals": {k: v for k, v in self.dual_lci_cuts.items()},
            "lci_node_alphas": {k: v for k, v in self.active_lci_node_alphas.items()},
            # Source arc per LCI cut: None for node-capacity cuts, (i,j) for arc-saturation
            # cuts.  The pricing DP uses this to gate the dual on the exact arc traversal.
            "lci_arcs": {k: v for k, v in self.active_lci_arcs.items()},
        }
        return result

    def get_node_visitation(self) -> Dict[int, float]:
        """
        Aggregate fractional node-visitation values from the current LP solution.

        For each node i, returns Σ_{k: i ∈ route_k} λ_k, i.e. the total
        fractional "coverage" of that node across all selected routes.

        Returns:
            Mapping from node ID to aggregated λ coverage, or empty dict if
            the model has not been solved yet.
        """
        if self.model is None:
            return {}

        try:
            node_visits: Dict[int, float] = {}
            for idx, var in enumerate(self.lambda_vars):
                val = var.X
                if val > 1e-6:
                    for node in self.routes[idx].node_coverage:
                        node_visits[node] = node_visits.get(node, 0.0) + val
            return node_visits
        except (gp.GurobiError, AttributeError):
            return {}

    def add_edge_clique_cut(
        self, u: int, v: int, coefficients: Optional[Dict[int, float]] = None, rhs: float = 1.0
    ) -> bool:
        """
        Add an Edge Clique cut specifically covering edge (u, v).

        Formulation:
            Σ_{k ∈ C} λ_k + Σ_{k ∈ \bar{C}} α_k λ_k <= |C| - 1

        Where C is a minimal cover of routes using edge (u, v), and α_k are
        lifting coefficients for routes not in the cover.

        Args:
            u, v: Endpoint nodes of the edge.
            coefficients: Mapping from route index to lifting coefficient α_k.
                If None, defaults to 1.0 for all routes using the edge.
            rhs: Right-hand side of the inequality (|C| - 1).

        Returns:
            True if the cut was added, False if it was redundant.
        """
        if self.model is None or not self.lambda_vars:
            return False

        edge_tuple = (min(u, v), max(u, v))
        key = edge_tuple

        if key in self.active_edge_clique_cuts:
            return False

        lhs = gp.LinExpr()
        found_columns = False
        for idx, route in enumerate(self.routes):
            # Check for edge (u, v) in route path
            nodes = [0] + route.nodes + [0]
            contains_edge = False
            for i in range(len(nodes) - 1):
                if tuple(sorted((nodes[i], nodes[i + 1]))) == edge_tuple:
                    contains_edge = True
                    break

            if contains_edge:
                coeff = 1.0
                if coefficients is not None:
                    coeff = coefficients.get(idx, 1.0)
                lhs.add(self.lambda_vars[idx], coeff)
                found_columns = True

        if not found_columns:
            return False

        constr = self.model.addConstr(lhs <= rhs, name=f"Edge_Clique_{edge_tuple[0]}_{edge_tuple[1]}")
        stored_coeffs = coefficients if coefficients is not None else {}
        self.active_edge_clique_cuts[key] = (constr, stored_coeffs)
        # Fix 6: Archive to global cut pool for cross-node reuse.
        self.global_cut_pool.add_cut("edge_clique", edge_tuple)
        self.model.update()
        return True

    def get_edge_usage(self) -> Dict[Tuple[int, int], float]:
        """
        Aggregate fractional edge visitation values from current LP solution.

        For an edge (i, j), returns Σ_{k: (i,j) ∈ route_k} λ_k.
        Edges are returned as canonical sorted tuples (min, max).

        Returns:
            Mapping from (u, v) -> fractional visitation sum.
        """
        return self._aggregate_edge_usage(only_elementary=False)

    def get_elementary_edge_usage(self) -> Dict[Tuple[int, int], float]:
        """
        Aggregate fractional edge visitation ONLY for strictly elementary routes.

        Exact separation engines (e.g., Rounded Capacity Cuts) mathematically
        require flow conservation on a support graph of elementary routes.
        This filter prevents cyclic ng-routes from breaking max-flow separation.
        """
        return self._aggregate_edge_usage(only_elementary=True)

    def _aggregate_edge_usage(self, only_elementary: bool) -> Dict[Tuple[int, int], float]:
        """Internal helper for edge usage aggregation with elementarity filter."""
        if self.model is None or not self.lambda_vars:
            return {}

        edge_usage: Dict[Tuple[int, int], float] = {}
        try:
            for idx, var in enumerate(self.lambda_vars):
                val = var.X
                if val > 1e-6:
                    route = self.routes[idx]

                    # Task 2: Filter cyclic support for exact separation
                    if only_elementary and len(set(route.nodes)) != len(route.nodes):
                        continue

                    nodes = [0] + route.nodes + [0]
                    for i in range(len(nodes) - 1):
                        u, v = nodes[i], nodes[i + 1]
                        edge = (min(u, v), max(u, v))
                        edge_usage[edge] = edge_usage.get(edge, 0.0) + val
            return edge_usage
        except Exception:
            return {}

    def add_subset_row_cut(self, node_set: Union[List[int], Set[int], FrozenSet[int]]) -> bool:
        """
        Add a 3-Subset Row Inequality (3-SRI) to the master problem.

        SRI Formulation:
            Σ_{k} ⌊ 1/2 * |S ∩ Route_k| ⌋ * λ_k <= 1

        For any subset S of 3 nodes, the sum of routes visiting at least 2 nodes
        in S cannot exceed 1. This strengthens the relaxation by cutting off
        fractional solutions where three routes visit pairs (node1, node2),
        (node2, node3), and (node1, node3) with value 0.5 each.

        Args:
            node_set: A list of exactly 3 customer nodes.

        Returns:
            True if the cut was successfully added to the model.
        """
        if self.model is None or len(node_set) != 3 or not self.lambda_vars:
            return False

        nodes = sorted(node_set)
        subset_frozenset = frozenset(nodes)
        if subset_frozenset in self.active_sri_cuts:
            return False

        cut_name = f"SRI_{nodes[0]}_{nodes[1]}_{nodes[2]}"

        lhs = gp.LinExpr()
        found_columns = False
        coeff_dict: Dict[str, float] = {}
        for idx, route in enumerate(self.routes):
            # Check node coverage for SRI violation calculation
            count = sum(1 for n in nodes if n in route.node_coverage)
            coeff = count // 2
            if coeff > 0:
                lhs.add(self.lambda_vars[idx], float(coeff))
                content = ",".join(map(str, route.nodes))
                route_h = hashlib.md5(content.encode()).hexdigest()
                coeff_dict[route_h] = float(coeff)
                found_columns = True

        if not found_columns:
            return False

        new_cut = self.model.addConstr(lhs <= 1.0, name=cut_name)
        self.active_sri_cuts[subset_frozenset] = new_cut
        # Archiving to Global Pool (SRIs are globally valid); store vector for orthogonality checks
        self.global_cut_pool.add_cut("sri", (subset_frozenset, coeff_dict))
        self.model.update()
        return True

    def add_capacity_cut(
        self,
        node_list: List[int],
        rhs: float,
        coefficients: Optional[Dict[int, float]] = None,
        is_global: bool = True,
        _skip_pool: bool = False,
    ) -> bool:
        """
        Add a Rounded Capacity Cut (RCC) to the master problem.

        The cut enforces that the number of edges crossing the boundary δ(S)
        of the node set S must be at least twice the minimum number of
        vehicles required to serve S.

        Constraint:  Σ_{k: route_k crosses δ(S)} crossings_k(S) * λ_k  >=  rhs

        Args:
            node_list: Nodes in set S.
            rhs: Right-hand side (2 * ⌈demand(S) / Q⌉).

        Returns:
            True if the cut was newly added, False if it already exists/failed.
        """
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_capacity_cuts:
            # Already have this cut
            return False

        # Build column-based representation of the cut
        lhs = gp.LinExpr()
        for idx, route in enumerate(self.routes):
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                lhs += float(crossings) * self.lambda_vars[idx]

        name = f"capacity_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        self.active_capacity_cuts[node_set] = constr
        if is_global and not _skip_pool:
            # Archive with the actual RHS so descendant nodes replay the correct
            # strength (2*ceil(demand(S)/Q)), not a hard-coded placeholder.
            self.global_cut_pool.add_cut("rcc", (node_set, rhs))

        if self.model is not None:
            self.model.update()
        return True

    def add_lci_cut(
        self,
        node_list: List[int],
        rhs: float,
        coefficients: Dict[int, float],
        node_alphas: Optional[Dict[int, float]] = None,
        arc: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Add a Lifted Cover Inequality (LCI) to the master problem.

        LCI formulation (Barnhart et al. 2000, §4):
            Σ_{k ∈ C} λ_k + Σ_{k ∉ C} α_k λ_k  ≤  |C| - 1

        LCI cuts are stored in `active_lci_cuts` (separate from `active_capacity_cuts`
        which holds >= RCC constraints) so that dual extraction uses the correct
        sign: Pi >= 0 for a <= constraint in a MAX LP.

        The optional ``node_alphas`` mapping (node_id → lifting coefficient α_i)
        enables the pricing subproblem to apply the correct per-node penalty when
        the cut's dual variable γ is nonzero.  Per Barnhart et al. (2000) §4.2:

            c'_lm^k = c_lm^k + π_lm + α_lm^k · γ_lm

        The optional ``arc`` parameter identifies the specific saturated arc (i, j)
        that generated this LCI (used by SaturatedArcLCIEngine).  When set, the
        pricing DP applies the dual **only** when traversing arc (i, j), matching
        the paper's arc-level formula exactly.  For node/capacity LCI (e.g.
        PhysicalCapacityLCIEngine) leave ``arc=None``; the DP then falls back to
        the node-visit approximation.

        Args:
            node_list: Customer nodes defining the cover set S.
            rhs: Right-hand side of the inequality (|C| - 1 or lifting bound K).
            coefficients: Route-index → route-level lifting coefficient (α_k).
            node_alphas: Node-id → per-node lifting coefficient (α_i).
                         If None, defaults to 1.0 for cover nodes, 0.0 for others.
            arc: Optional source arc (i, j) for arc-saturation LCI.
        """
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_lci_cuts:
            return False

        lhs = gp.LinExpr()
        for idx, coeff in coefficients.items():
            if idx < len(self.lambda_vars):
                lhs += coeff * self.lambda_vars[idx]

        if lhs.size() == 0:
            return False

        name = f"lci_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs <= rhs, name=name)
        self.active_lci_cuts[node_set] = constr
        # Store per-node alphas for pricing dual integration.
        effective_node_alphas: Dict[int, float] = node_alphas if node_alphas is not None else {}
        self.active_lci_node_alphas[node_set] = effective_node_alphas
        # Store source arc (None for node/capacity LCI, set for arc-saturation LCI).
        self.active_lci_arcs[node_set] = arc
        # Archive with rhs, route coefficients, node_alphas, AND arc for descendant replay.
        self.global_cut_pool.add_cut("lci", (node_set, rhs, coefficients, effective_node_alphas, arc))
        if self.model is not None:
            self.model.update()
        return True

    def add_sec_cut(
        self,
        node_list: Union[List[int], Set[int], FrozenSet[int]],
        rhs: float,
        cut_name: str = "",
        global_cut: bool = True,
        node_i: int = -1,
        node_j: int = -1,
        facet_form: str = "2.1",
    ) -> bool:
        """
        Add a Subtour Elimination Cut (SEC) or PC-SEC to the master problem.

        Args:
            node_list: Set of nodes in the subtour.
            rhs: Right-hand side value.
            cut_name: Optional name for the constraint.
            global_cut: If True, the cut is stored in the global registry.
            node_i: Index of node i for Form 2.2 and 2.3 PC-SECs.
            node_j: Index of node j for Form 2.3 PC-SECs.
            facet_form: Form indicator for PC-SECs (e.g. "2.1", "2.3").
        """
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        registry = self.active_sec_cuts if global_cut else self.active_sec_cuts_local

        if node_set in registry:
            return False

        lhs = gp.LinExpr()
        for idx, route in enumerate(self.routes):
            # Task 7: PC-SEC Form 2.3 Formulation Fix.
            # We calculate the coefficient C_k for route k in the cut Σ C_k λ_k >= RHS.
            # Form 2.3: Σ x_e >= 2(y_i + y_j - 1)  =>  Σ x_e - 2y_i - 2y_j >= -2.
            # C_k = crossings(route_k, S) - 2 * (1 if i in route_k) - 2 * (1 if j in route_k).
            val = float(self._count_crossings(route, node_set))

            if node_i > 0 and node_i in route.node_coverage:
                val -= 2.0
            if node_j > 0 and node_j in route.node_coverage:
                val -= 2.0

            if abs(val) > 1e-6:
                lhs += val * self.lambda_vars[idx]

        name = cut_name if cut_name else f"sec_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        registry[node_set] = constr

        # Archiving to Global Pool
        if global_cut and node_i < 0 and node_j < 0:
            # Form 2.1 is always valid Σ x_e >= 2
            self.global_cut_pool.add_cut("sec_2.1", node_set)

        self.model.update()
        return True

    def remove_local_cuts(self) -> int:
        """
        Delete all node-local cuts from the Gurobi model and clear registries.

        Returns:
            Number of cuts removed.
        """
        if self.model is None:
            return 0
        removed = 0
        for constr in self.active_sec_cuts_local.values():
            try:
                self.model.remove(constr)
                removed += 1
            except gp.GurobiError:
                continue

        self.active_sec_cuts_local.clear()
        self.dual_sec_cuts_local = {}
        self.model.update()
        return removed

    def _count_crossings(self, route: Route, node_set: FrozenSet[int]) -> int:
        """Count how many times a route crosses the boundary δ(S)."""
        crossings = 0
        path_nodes = [0] + route.nodes + [0]
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if (u in node_set) != (v in node_set):
                crossings += 1
        return crossings

    def has_artificial_variables_active(self, tol: float = 1e-6) -> bool:
        """
        Check whether any artificial variable is non-zero in the current solution.

        A non-zero α_i after solving the IP indicates that node i cannot be
        covered by the current column pool under the active branching constraints,
        so the B&B node should be declared infeasible and pruned.

        Args:
            tol: Threshold below which a variable value is considered zero.

        Returns:
            True if at least one α_i > tol, indicating infeasibility.
        """
        # artificial_vars have been removed; infeasibility is now signalled
        # by GRB.INFEASIBLE status directly.
        if self.model is None:
            return False
        return self.model.Status == GRB.INFEASIBLE

    def save_basis(self) -> Optional[Tuple[List[int], List[int]]]:
        """Save the basis status (VBasis, CBasis) for all variables and constraints."""
        if self.model is None or self.model.Status != GRB.OPTIMAL:
            return None
        try:
            vbasis = [v.VBasis for v in self.model.getVars()]
            cbasis = [c.CBasis for c in self.model.getConstrs()]
            return (vbasis, cbasis)
        except Exception:
            return None

    def restore_basis(self, vbasis: List[int], cbasis: List[int]) -> None:
        """
        Restore the basis to warm-start the simplex algorithm.
        """
        if self.model is None or vbasis is None or cbasis is None:
            return

        # Partial restore robust to column-pool mismatch
        vars_ = self.model.getVars()
        constrs_ = self.model.getConstrs()

        n_vars_saved = len(vbasis)
        n_constrs_saved = len(cbasis)
        n_vars_now = len(vars_)
        n_constrs_now = len(constrs_)

        if n_vars_now != n_vars_saved or n_constrs_now != n_constrs_saved:
            logger.debug(
                f"restore_basis: size mismatch — saved ({n_vars_saved} vars, "
                f"{n_constrs_saved} constrs) vs current ({n_vars_now} vars, "
                f"{n_constrs_now} constrs). Partial restore; warm-start may be degraded."
            )

        n_vars = min(n_vars_saved, n_vars_now)
        n_constrs = min(n_constrs_saved, n_constrs_now)

        for v, b in zip(vars_[:n_vars], vbasis[:n_vars]):
            v.VBasis = b
        for c, b in zip(constrs_[:n_constrs], cbasis[:n_constrs]):
            c.CBasis = b

        self.model.update()
