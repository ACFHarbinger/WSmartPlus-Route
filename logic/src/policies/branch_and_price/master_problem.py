"""
Master Problem for Branch-and-Price VRPP.

Implements the set covering formulation where each column represents a feasible route.
The master problem selects routes to cover all mandatory nodes while maximising profit.

Based on Section 3.2 of Barnhart et al. (1998).
"""

from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB


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

    def __repr__(self) -> str:
        return f"Route(nodes={self.nodes}, profit={self.profit:.2f})"


class VRPPMasterProblem:
    """
    Set Covering Master Problem for VRPP with Column Generation.

    This is a pure Branch-and-Price formulation WITHOUT cutting planes.
    All capacity and routing constraints are handled implicitly by the ESPPRC
    pricing subproblem, which only generates feasible routes.

    Formulation (Barnhart et al. 1998):
        max  Σ_k profit_k * λ_k  -  M * Σ_{i ∈ mand} α_i
        s.t. Σ_k a_{ik} * λ_k + α_i  >=  1   ∀ i ∈ mandatory  (set covering)
             Σ_k a_{ik} * λ_k         <=  1   ∀ i ∈ optional   (packing)
             Σ_k λ_k                  <=  K                     (fleet limit, optional)
             λ_k ∈ {0, 1}             ∀ k                       (route selection)
             α_i >= 0                 ∀ i ∈ mandatory           (artificial var)

    Key design decisions (Barnhart et al. 1998):
        Set Covering (>= 1) vs Set Partitioning (== 1):
            Using >= 1 for mandatory-node coverage constraints improves LP dual
            stability across B&B nodes.  A strict == 1 constraint produces dual
            values that can flip sign at every branching step, causing column
            generation to stall.  The >= relaxation keeps duals non-negative and
            monotone, which accelerates convergence of the pricing subproblem.

        Artificial variables (Big-M):
            Adding α_i >= 0 with objective coefficient -M (≫ 0) to each mandatory
            node's coverage constraint guarantees LP feasibility at *every* node of
            the B&B tree, even when the current column pool cannot cover all
            mandatory nodes.  In the optimal solution M drives α_i to zero; if any
            α_i > 0 in the final solution, the node is declared infeasible and pruned.

        Dynamic cuts (Rounded Capacity Cuts):
            RCC cuts can be added to the master via add_set_packing_capacity_cut().
            Their dual variables (π_S) are extracted after each LP solve and stored in
            dual_capacity_cuts. This follows the combined column-and-row generation
            approach of Barnhart et al. (1998, Section 7) and the capacity cut
            formulation of Desrochers, Desrosiers, and Solomon (1992). After adding
            a cut, the LP is reoptimised and the pricing problem is re-run with
            modified arc costs that account for the new dual variable.

            Comb inequalities and Lifted Cover Inequalities are NOT added to this master
            because their dual integration into the RCSPP state-space is non-trivial and
            currently unsupported (see cutting_planes.py for details).
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
    ) -> None:
        """
        Initialise the master problem.

        Args:
            n_nodes: Number of customer nodes (excluding depot, depot = index 0).
            mandatory_nodes: Set of node indices that must be visited.
            cost_matrix: Distance matrix of shape (n_nodes+1, n_nodes+1);
                index 0 is the depot.
            wastes: Mapping from node ID to waste volume (kg).
            capacity: Vehicle payload capacity (kg).
            revenue_per_kg: Revenue earned per kg of waste collected.
            cost_per_km: Operating cost per km of travel.
            vehicle_limit: Maximum number of vehicles (routes) allowed, or
                None to impose no fleet-size constraint.
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

        # Task 5 & Part 5 Task 4: Dynamic Big-M for Numerical Stability
        # Penalty must strictly exceed the max possible profit of ANY single route.
        max_single_node_revenue = max(self.wastes.values(), default=0.0) * self.R
        # A route visits at most n_nodes customers.
        max_route_profit = max_single_node_revenue * self.n_nodes
        self.BIG_M = max(1000.0, 2.0 * max_route_profit)

        self.routes: List[Route] = []

        # Gurobi model
        self.model: Optional[gp.Model] = None
        self.lambda_vars: List[gp.Var] = []

        # Artificial variables α_i for mandatory nodes (Big-M penalty).
        # These are added once in build_model() and never removed, ensuring
        # LP feasibility is preserved throughout the entire B&B tree.
        self.artificial_vars: Dict[int, gp.Var] = {}

        self.dual_node_coverage: Dict[int, float] = {}
        self.dual_vehicle_limit: float = 0.0

        # Map from FrozenSet[int] (node set S) to Gurobi constraint.
        self.active_capacity_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        # Dual values π_S used by pricing subproblem reduced-cost calc.
        self.dual_capacity_cuts: Dict[FrozenSet[int], float] = {}

        # ---- SEC Cut Tracking (PC-SEC) ------------------------------------
        self.active_sec_cuts: Dict[FrozenSet[int], gp.Constr] = {}
        self.dual_sec_cuts: Dict[FrozenSet[int], float] = {}

    def remove_unpromising_columns(self, rc_floor: float) -> int:
        """
        Remove columns with reduced cost below rc_floor from the model.

        Frees up memory and accelerates LP solves by keeping the model small.
        Only non-basic variables (X=0) are candidates for deletion.

        Args:
            rc_floor: Reduced cost lower bound (e.g., -100.0). Variables with
                RC < rc_floor are candidates for removal.

        Returns:
            Number of columns removed.
        """
        if self.model is None or not self.lambda_vars:
            return 0

        # Identify indices to remove. We check RC (reduced cost) and X (value).
        # In a maximization LP, non-basic variables have RC <= 0.
        to_remove: List[int] = []
        for i, var in enumerate(self.lambda_vars):
            try:
                # var.RC is the reduced cost in Gurobi.
                # var.X is the value (0.0 for non-basic columns in most cases).
                if var.X < 1e-6 and rc_floor > var.RC:
                    to_remove.append(i)
            except (gp.GurobiError, AttributeError):
                # RC or X might not be available if the model wasn't solved optimally
                continue

        if not to_remove:
            return 0

        # Remove columns from Gurobi and local pool.
        # We must delete in reverse index order to maintain valid indexing.
        for i in sorted(to_remove, reverse=True):
            self.model.remove(self.lambda_vars[i])
            self.lambda_vars.pop(i)
            self.routes.pop(i)

        self.model.update()
        return len(to_remove)

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

    def _add_column_to_model(self, route: Route) -> None:
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

        # Wire into active capacity cuts if any exist
        for node_set, constr in self.active_capacity_cuts.items():
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                self.model.chgCoeff(constr, var, float(crossings))

        # Wire into active SEC cuts if any exist
        for node_set, constr in self.active_sec_cuts.items():
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                self.model.chgCoeff(constr, var, float(crossings))

        self.model.update()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self, initial_routes: Optional[List[Route]] = None) -> None:
        """
        Build the Gurobi model for the set-covering master problem.

        Implements the Big-M / artificial-variable formulation of
        Barnhart et al. (1998):

        1. Mandatory-node coverage constraints use >= 1 (Set Covering) so that
           dual values remain non-negative and stable across B&B nodes.
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
        # Use the Barrier (Interior Point) algorithm to solve the LP relaxation.
        # This finds a central optimal dual solution, preventing the "bang-bang"
        # extreme-point oscillation typical in degenerate set covering problems.
        # Reference: Barnhart et al. (1998).
        self.model.Params.Method = 2
        self.lambda_vars = []
        self.artificial_vars = {}

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

        # ---- Artificial (α) variables for mandatory nodes ----------------
        # Each α_i is a non-negative continuous variable penalised with -BIG_M
        # in the maximisation objective, so the solver is strongly incentivised
        # to drive them to zero by selecting routes that cover node i.
        for node in self.mandatory_nodes:
            self.artificial_vars[node] = self.model.addVar(
                obj=-self.BIG_M,
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                name=f"alpha_{node}",
            )

        # ---- Mandatory-node coverage constraints (Set Covering: >= 1) ----
        # Using >= 1 instead of == 1:
        #   * Dual values (shadow prices) of >= constraints in a MAX LP are
        #     non-positive; taking their absolute value gives a non-negative
        #     contribution to reduced costs, which is the sign convention
        #     expected by the pricing subproblem.
        #   * The relaxation allows "over-coverage" (visiting a node on more
        #     than one route), which is harmless because the objective drives
        #     the solver towards the most profitable non-redundant solution.
        for node in self.mandatory_nodes:
            lhs = gp.LinExpr()
            for idx, route in enumerate(self.routes):
                if node in route.node_coverage:
                    lhs += self.lambda_vars[idx]
            # Artificial variable ensures constraint is always feasible.
            lhs += self.artificial_vars[node]

            self.model.addConstr(
                lhs >= 1.0,
                name=f"coverage_{node}",
            )

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

    def solve_lp_relaxation(self) -> Tuple[float, Dict[int, float]]:
        """
        Solve the LP relaxation of the master problem and extract dual values.

        The integrality constraints on all λ variables are temporarily relaxed
        to continuous [0, 1] before solving.  Artificial variables are already
        continuous, so they require no modification.

        Returns:
            Tuple of:
                obj_value  – LP relaxation objective value.
                route_values – Mapping {route_index: λ_k value}.

        Raises:
            ValueError: If build_model() has not been called.
            RuntimeError: If the LP solve does not terminate optimally.
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

        if self.model.Status != GRB.OPTIMAL:
            return 0.0, {}

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
                    # >= 1 constraint in MAX LP: Pi <= 0, dual value = -Pi >= 0.
                    self.dual_node_coverage[node] = max(0.0, -constr.Pi)
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
                try:
                    # <= 1 constraint in MAX LP: Pi <= 0, dual value = -Pi >= 0.
                    self.dual_vehicle_limit = max(0.0, -constr.Pi)
                except gp.GurobiError:
                    pass

        # ---- Extract duals for capacity cuts -----------------------------
        self.dual_capacity_cuts = {}
        if self.model is not None and self.model.Status == GRB.OPTIMAL:
            for node_set, constr in self.active_capacity_cuts.items():
                try:
                    # Dual π_S for a >= constraint in MAX LP is non-positive.
                    self.dual_capacity_cuts[node_set] = max(0.0, -constr.Pi)
                except (gp.GurobiError, AttributeError):
                    self.dual_capacity_cuts[node_set] = 0.0

        # ---- Extract duals for SEC cuts -----------------------------
        self.dual_sec_cuts = {}
        if self.model is not None and self.model.Status == GRB.OPTIMAL:
            for node_set, constr in self.active_sec_cuts.items():
                try:
                    self.dual_sec_cuts[node_set] = max(0.0, -constr.Pi)
                except (gp.GurobiError, AttributeError):
                    self.dual_sec_cuts[node_set] = 0.0

        return obj_value, route_values

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

    def get_reduced_cost_coefficients(self) -> Dict[Union[int, str], float]:
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
        duals = self.dual_node_coverage.copy()
        if self.vehicle_limit is not None:
            duals["vehicle_limit"] = self.dual_vehicle_limit  # type: ignore[index]
        return duals  # type: ignore[return-value]

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
        except Exception:
            return {}

    def get_edge_usage(self) -> Dict[Tuple[int, int], float]:
        """
        Aggregate fractional edge visitation values from current LP solution.

        For an edge (i, j), returns Σ_{k: (i,j) ∈ route_k} λ_k.
        Edges are returned as canonical sorted tuples (min, max).

        Returns:
            Mapping from (u, v) -> fractional visitation sum.
        """
        if self.model is None or not self.lambda_vars:
            return {}

        edge_usage: Dict[Tuple[int, int], float] = {}
        try:
            for idx, var in enumerate(self.lambda_vars):
                val = var.X
                if val > 1e-6:
                    route = self.routes[idx]
                    # Route starts and ends at depot (0)
                    nodes = [0] + route.nodes + [0]
                    for i in range(len(nodes) - 1):
                        u, v = nodes[i], nodes[i + 1]
                        edge = (min(u, v), max(u, v))
                        edge_usage[edge] = edge_usage.get(edge, 0.0) + val
            return edge_usage
        except Exception:
            return {}

    def add_set_packing_capacity_cut(self, node_list: List[int], rhs: float) -> bool:
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

        name = f"rcc_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        self.active_capacity_cuts[node_set] = constr
        self.model.update()
        return True

    def add_sec_cut(self, node_list: List[int], rhs: float, cut_name: str = "") -> bool:
        """
        Add a Subtour Elimination Constraint (SEC / PC-SEC) to the master problem.

        Constraint: Σ_{k: route_k crosses δ(S)} crossings_k(S) * λ_k >= rhs

        Algorithmic Alignment:
            *   This method iterates over self.routes to wire EXISTING columns into the
                new constraint.
            *   Columns added after this call (e.g. in the next Phase 1 of the CG loop)
                are wired via _add_column_to_model.
            *   Phase 1 (Column Generation) and Phase 2 (Cut Separation) are non-interleaving
                within a single outer iteration, ensuring consistency.

        The dual of this constraint is tracked in dual_sec_cuts so that it can be
        passed to the pricing subproblem's reduced-cost calculation.

        Args:
            node_list: Customer nodes in the cut set S.
            rhs: Right-hand side value (2.0 for Form 2.1, 2*y_i for Form 2.2, etc.).
            cut_name: Optional name suffix for Gurobi constraint naming.

        Returns:
            True if the cut was newly added, False if it already exists.
        """
        if self.model is None:
            return False

        node_set = frozenset(node_list)
        if node_set in self.active_sec_cuts:
            return False

        lhs = gp.LinExpr()
        for idx, route in enumerate(self.routes):
            crossings = self._count_crossings(route, node_set)
            if crossings > 0:
                lhs += float(crossings) * self.lambda_vars[idx]

        name = f"sec_{cut_name}_{abs(hash(node_set))}"
        constr = self.model.addConstr(lhs >= rhs, name=name)
        self.active_sec_cuts[node_set] = constr
        self.model.update()
        return True

    def find_and_add_violated_rcc(
        self,
        route_values: Dict[int, float],
        routes: List[Route],
        max_cuts: int = 5,
    ) -> int:
        """
        Separate and add Rounded Capacity Cuts (RCC) based on the current LP solution.

        This follows Section 7 of Barnhart et al. (1998) and Desrochers et al. (1992)
        using a connectivity heuristic:
        1. Build the fractional flow support graph (arcs with x_uv > 0).
        2. Identify connected components (S) of customer nodes.
        3. For each component S, check if the routing bound is violated:
           Σ_{k} x^k(δ(S)) λ_k  <  2 * ⌈ (Σ_{i∈S} waste_i) / Q ⌉
        """
        if not route_values or not routes:
            return 0

        # Aggregate arc flows from active routes
        arc_flow: Dict[Tuple[int, int], float] = {}
        for idx, val in route_values.items():
            if val < 1e-6 or idx >= len(routes):
                continue
            path = [0] + routes[idx].nodes + [0]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                arc = (u, v)
                arc_flow[arc] = arc_flow.get(arc, 0.0) + val

        # Find connected components of customer nodes in the support graph (ignoring depot)
        components = self._find_customer_components(arc_flow)

        # Evaluate each component S as a cut candidate
        cuts_added = 0
        for S in components:
            if not S:
                continue
            total_waste = sum(self.wastes.get(i, 0.0) for i in S)
            rhs = 2.0 * np.ceil(total_waste / self.capacity) if self.capacity > 0 else 2.0

            # Calculate current LHS value (sum of flow crossing the boundary delta(S))
            lhs_val = sum(flow for (u, v), flow in arc_flow.items() if (u in S) != (v in S))

            # Add if violated by more than 1e-4
            if lhs_val < rhs - 1e-4 and self.add_set_packing_capacity_cut(list(S), rhs):
                cuts_added += 1
                if cuts_added >= max_cuts:
                    break
        return cuts_added

    def _find_customer_components(self, arc_flow: Dict[Tuple[int, int], float]) -> List[Set[int]]:
        """Identify connected components of customer nodes in the support graph."""
        adj: Dict[int, Set[int]] = {}
        customer_nodes = set()
        for u, v in arc_flow.keys():
            if u != 0 and v != 0:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
                customer_nodes.add(u)
                customer_nodes.add(v)

        visited = set()
        components: List[Set[int]] = []
        for node in customer_nodes:
            if node not in visited:
                comp = set()
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        comp.add(curr)
                        stack.extend(adj.get(curr, set()) - visited)
                components.append(comp)
        return components

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
        if self.model is None:
            return False
        try:
            return any(tol < alpha.X for alpha in self.artificial_vars.values())
        except Exception:
            return False

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

    def restore_basis(self, basis: Optional[Tuple[List[int], List[int]]]) -> None:
        """Restore basis status to accelerate the next LP solve."""
        if basis is None or self.model is None:
            return
        vbasis, cbasis = basis
        vars = self.model.getVars()
        constrs = self.model.getConstrs()
        if len(vbasis) == len(vars) and len(cbasis) == len(constrs):
            for v, b in zip(vars, vbasis):
                v.VBasis = b
            for c, b in zip(constrs, cbasis):
                c.CBasis = b
            self.model.update()
