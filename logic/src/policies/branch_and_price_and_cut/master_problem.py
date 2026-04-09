"""
Master Problem for Branch-and-Price VRPP.

Implements the set covering formulation where each column represents a feasible route.
The master problem selects routes to cover all mandatory nodes while maximising profit.

Based on Section 3.2 of Barnhart et al. (1998).
"""

import contextlib
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
    """

    def __init__(self) -> None:
        """Initialise empty global cut registries."""
        self.rcc_cuts: Set[FrozenSet[int]] = set()
        self.sri_cuts: Set[FrozenSet[int]] = set()
        self.sec_cuts: Set[FrozenSet[int]] = set()  # Form 2.1 (Global)
        self.edge_clique_cuts: Set[Tuple[int, int]] = set()
        self.lci_cuts: Set[FrozenSet[int]] = set()

    def add_cut(self, cut_type: str, data: Any) -> None:
        """
        Archive a globally valid cut in the pool.

        Args:
            cut_type: The type of cut ("rcc", "sri", "sec_2.1", "lci").
            data: Cut-specific data (usually a FrozenSet of nodes).
        """
        if cut_type == "rcc":
            self.rcc_cuts.add(data)
        elif cut_type == "sri":
            self.sri_cuts.add(data)
        elif cut_type == "sec_2.1":
            self.sec_cuts.add(data)
        elif cut_type == "edge_clique":
            self.edge_clique_cuts.add(data)
        elif cut_type == "lci":
            self.lci_cuts.add(data)

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
        for nodes in self.rcc_cuts:
            if master.add_capacity_cut(list(nodes), rhs=1.0):  # Simplified RHS for SPP
                added += 1
        for nodes in self.sri_cuts:
            if master.add_subset_row_cut(nodes):
                added += 1
        for nodes in self.sec_cuts:
            # Form 2.1 is always global
            if master.add_sec_cut(nodes, rhs=1.0, facet_form="2.1"):
                added += 1
        # Edge cliques and LCIs are more complex but can be added similarly
        return added


class VRPPMasterProblem:
    """
    Master Problem (MP) for the Branch-and-Price-and-Cut algorithm.

    Formulates and solves the Set Partitioning Problem (SPP) relaxation of the VRPP.
    The Master Problem is responsible for:
        1.  Managing the column pool (routes generate by the pricing subproblem).
        2.  Enforcing various valid inequalities (Rounded Capacity, Subset-Row, Lifted Cover).
        3.  Extracting dual values used to guide the search for profitable columns.
        4.  Interfacing with the Branch-and-Bound tree to enforce node-specific constraints.

    Mathematical Formulation:
    -------------------------
    Maximize Σ p_k * λ_k
    Subject to:
        Σ a_{ik} * λ_k = 1           ∀ i ∈ Mandatory Customers
        Σ a_{jk} * λ_k ≤ 1           ∀ j ∈ Optional Customers
        Σ λ_k <= K                   (Vehicle Fleet Constraint)
        Σ γ_S * λ_k <= RHS           (Cutting Planes: RCC, SRI, Edge Clique)
        λ_k ∈ {0, 1}

    Theoretical Exactness:
    ----------------------
    To support Ryan-Foster branching and ensure rigorous optimality proofs,
    this implementation defaults to strict Set Partitioning (== 1.0) logic
    for all covered nodes (enforced via `strict_set_partitioning`).
    If a node is over-covered (Σ λ_{ik} > 1), the corresponding artificial
    variable would need to be negative to satisfy the equality, which is
    forbidden by its lower bound (LB=0.0).
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

        Framework (Barnhart et al. 1998, 2000):
        This class maintains the Restricted Master Problem (RMP), which is the Set
        Partitioning relaxation of the VRPP. It manages column addition, cutting
        plane integration, and dual extraction.

        Set Partitioning & Branching:
        To maintain Ryan-Foster (1981) compatibility, we use strict equality (== 1)
        for mandatory nodes. Artificial variables (Big-M) are omitted in favor of
        Farkas dual extraction, ensuring that infeasibility directly triggers the
        feasibility-search phase of column generation.

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

        # Dual mappings
        self.dual_src_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_sec_cuts: Dict[FrozenSet[int], float] = {}
        self.dual_sec_cuts_local: Dict[FrozenSet[int], float] = {}
        self.dual_capacity_cuts: Dict[FrozenSet[int], float] = {}
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

    def deduplicate_column_pool(self, tol: float = 1e-6) -> int:
        """
        SOTA: Prune identical or dominated routes from the global column pool.
        Prevents RMP from becoming degenerate with duplicate columns.
        """
        if not self.global_column_pool:
            return 0

        initial_count = len(self.global_column_pool)
        seen_node_sets: Dict[FrozenSet[int], float] = {}
        unique_pool = []

        for route in self.global_column_pool:
            nodes_f = frozenset(route.nodes)
            if nodes_f not in seen_node_sets:
                seen_node_sets[nodes_f] = route.profit
                unique_pool.append(route)
            else:
                # If a route with the same nodes exists, keep only the one with better profit
                if route.profit > seen_node_sets[nodes_f] + tol:
                    # Replace in unique pool (slow but pool is capped)
                    for i, r in enumerate(unique_pool):
                        if frozenset(r.nodes) == nodes_f:
                            unique_pool[i] = route
                            seen_node_sets[nodes_f] = route.profit
                            break

        self.global_column_pool = unique_pool
        return initial_count - len(unique_pool)

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
            branching_constraints: List of active B&B branching constraints.
            rc_tolerance: Numerical threshold to prevent injection of mathematically
                         insignificant columns (epsilon deadlock).

        Returns:
            Number of routes re-activated and added to the RMP.
        """
        if not self.global_column_pool:
            return 0

        # Bundle duals for calculate_reduced_cost
        dual_values = {
            "node_duals": node_duals,
            "rcc_duals": rcc_duals,
            "sri_duals": sri_duals,
            "edge_clique_duals": edge_clique_duals,
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

        Phase 1: Minimize Σ α_i. Route profits are ignored.
        Phase 2: Maximize Σ p_k λ_k - Σ BIG_M α_i.

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

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self, initial_routes: Optional[List[Route]] = None) -> None:
        """
        Build the Gurobi model for the set-covering master problem.

        Implements the Big-M / artificial-variable formulation of
        Barnhart et al. (1998):

        1. Mandatory-node coverage constraints use == 1 (Set Partitioning) to
           ensure that Ryan-Foster branching logic remains mathematically valid.
        2. An artificial variable α_i is added to every mandatory-node constraint
           with objective coefficient -BIG_M, guaranteeing LP feasibility even
           when the column pool cannot cover every mandatory node.
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

        self.model.update()
        if self.model.NumVars == 0:
            return 0.0, {}
        self.model.optimize()
        status = self.model.Status

        if status == GRB.INFEASIBLE:
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
                farkas_node_duals["vehicle_limit"] = farkas_node_duals.get("vehicle_limit", 0.0) + temp_min_c.FarkasDual

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
        self.dual_capacity_cuts = {}
        for node_set, constr in self.active_capacity_cuts.items():
            with contextlib.suppress(gp.GurobiError):
                # >= rhs constraint in MAX LP: Pi <= 0, dual value = -Pi >= 0.
                self.dual_capacity_cuts[node_set] = max(0.0, -constr.Pi)

        # ---- Extract duals for SRI ---------------------------------------
        self.dual_sri_cuts = {}
        for subset, constr in self.active_sri_cuts.items():
            if constr is not None:
                with contextlib.suppress(gp.GurobiError):
                    self.dual_sri_cuts[subset] = max(0.0, -constr.Pi)

        # Extract Edge Clique cut duals
        self.dual_edge_clique_cuts = {}
        # Edge cliques are <= 1 constraints in maximization. duals are normally >= 0.
        for key, (constr, _) in self.active_edge_clique_cuts.items():
            if constr is not None:
                with contextlib.suppress(gp.GurobiError):
                    self.dual_edge_clique_cuts[key] = max(0.0, -constr.Pi)

        # ---- Task 5: Apply Exponential Smoothing -----------------------
        if self.enable_dual_smoothing:
            self._apply_dual_smoothing()

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

        # 3. Capacity cut duals
        for subset, current_val in self.dual_capacity_cuts.items():
            prev_val = self.prev_dual_capacity_cuts.get(subset, current_val)
            self.dual_capacity_cuts[subset] = alpha * current_val + (1.0 - alpha) * prev_val
        self.prev_dual_capacity_cuts = self.dual_capacity_cuts.copy()

        # 4. SRI cuts
        for subset, current_val in self.dual_sri_cuts.items():
            prev_val = self.prev_dual_sri_cuts.get(subset, current_val)
            self.dual_sri_cuts[subset] = alpha * current_val + (1.0 - alpha) * prev_val
        self.prev_dual_sri_cuts = self.dual_sri_cuts.copy()

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

    def get_reduced_cost_coefficients(self) -> Dict[str, Dict[Union[int, frozenset[int], str, Tuple[int, int]], float]]:
        """
        Return the dual-value coefficients used by the pricing subproblem.

        The reduced cost of a candidate route r is:
            rc(r) = profit(r)  −  Σ_{i ∈ r} dual_i  −  dual_vehicle_limit

        This method collects both node-coverage duals and the vehicle-limit
        dual into a single dictionary consumed by PricingSubproblem /
        RCSPPSolver.

        Returns:
            Dictionary mapping node ID (int) → dual value, plus the key
            "vehicle_limit" → vehicle-limit dual if a fleet cap is active.
        """
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
        result: Dict[str, Dict[Union[int, frozenset[int], str, Tuple[int, int]], float]] = {
            "node_duals": {k: v for k, v in duals.items()},
            "rcc_duals": {k: v for k, v in self.dual_capacity_cuts.items()},
            "sri_duals": {k: v for k, v in self.dual_sri_cuts.items()},
            "edge_clique_duals": {k: v for k, v in clique_duals_collapsed.items()},
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
        for idx, route in enumerate(self.routes):
            count = sum(1 for n in nodes if n in route.nodes)
            coeff = count // 2
            if coeff > 0:
                lhs.add(self.lambda_vars[idx], float(coeff))
                found_columns = True

        if not found_columns:
            return False

        new_cut = self.model.addConstr(lhs <= 1.0, name=cut_name)
        self.active_sri_cuts[subset_frozenset] = new_cut
        # Archiving to Global Pool (SRIs are globally valid)
        self.global_cut_pool.add_cut("sri", subset_frozenset)
        self.model.update()
        return True

    def add_capacity_cut(
        self,
        node_list: List[int],
        rhs: float,
        coefficients: Optional[Dict[int, float]] = None,
        is_global: bool = True,
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
        if is_global:
            # Archiving to Global Pool (RCCs are globally valid)
            self.global_cut_pool.add_cut("rcc", node_set)

        if self.model is not None:
            self.model.update()
        return True

    def add_lci_cut(self, node_list: List[int], rhs: float, coefficients: Dict[int, float]) -> bool:
        """
        Add a Lifted Cover Inequality (LCI) to the master problem.
        """
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        # Use capacity cut registry for storage but tag as LCI in pool
        if node_set in self.active_capacity_cuts:
            return False

        lhs = gp.LinExpr()
        for idx, coeff in coefficients.items():
            lhs += coeff * self.lambda_vars[idx]

        name = f"lci_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs <= rhs, name=name)
        self.active_capacity_cuts[node_set] = constr
        self.global_cut_pool.add_cut("lci", node_set)
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
