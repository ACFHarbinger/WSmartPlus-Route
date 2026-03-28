"""
Master Problem for Branch-and-Price VRPP.

Implements the set covering formulation where each column represents a feasible route.
The master problem selects routes to cover all mandatory nodes while maximising profit.

Based on Section 3.2 of Barnhart et al. (1998).
"""

from typing import Dict, List, Optional, Set, Tuple, Union

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

        No dynamic cuts:
            Capacity and routing feasibility are delegated entirely to the ESPPRC
            pricing subproblem (rcspp_dp.py).  Adding cutting planes to the master
            would introduce extra dual variables that the pricing DP cannot account
            for, creating a dual desync and invalidating the reduced-cost criterion.
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
        self.depot = 0

        # Column pool
        self.routes: List[Route] = []

        # Gurobi model
        self.model: Optional[gp.Model] = None
        self.lambda_vars: List[gp.Var] = []

        # Artificial variables α_i for mandatory nodes (Big-M penalty).
        # These are added once in build_model() and never removed, ensuring
        # LP feasibility is preserved throughout the entire B&B tree.
        self.artificial_vars: Dict[int, gp.Var] = {}

        # Big-M penalty applied to artificial variables in the objective.
        # Must be strictly larger than the maximum achievable profit to ensure
        # that artificial variables are zero in any non-infeasible solution.
        self.BIG_M: float = 1e6

        # Dual values extracted after each LP solve (used by pricing subproblem)
        self.dual_node_coverage: Dict[int, float] = {}
        self.dual_vehicle_limit: float = 0.0

    # ------------------------------------------------------------------
    # Column management
    # ------------------------------------------------------------------

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
            alpha = self.model.addVar(
                obj=-self.BIG_M,  # Penalise artificial variable (maximisation)
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                name=f"artificial_{node}",
            )
            self.artificial_vars[node] = alpha

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
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"LP relaxation failed with status {self.model.Status}")

        obj_value = self.model.ObjVal
        route_values = {idx: var.X for idx, var in enumerate(self.lambda_vars)}

        # ---- Extract dual values -----------------------------------------
        # For a >= constraint in a MAX LP, Gurobi reports a non-positive π.
        # The pricing subproblem expects non-negative duals (interpreted as the
        # "value" of covering a node), so we store |π|.
        self.dual_node_coverage = {}
        for node in range(1, self.n_nodes + 1):
            constr = self.model.getConstrByName(f"coverage_{node}")
            if constr is not None:
                self.dual_node_coverage[node] = abs(constr.Pi)

        # For the <= vehicle-limit constraint in a MAX LP, the dual is the
        # opportunity cost of using one more vehicle; clamp to non-negative.
        self.dual_vehicle_limit = 0.0
        if self.vehicle_limit is not None:
            constr = self.model.getConstrByName("vehicle_limit")
            if constr is not None:
                self.dual_vehicle_limit = max(0.0, constr.Pi)

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

        for var in self.lambda_vars:
            var.VType = GRB.BINARY

        self.model.update()
        self.model.optimize()

        if self.model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            raise RuntimeError(f"IP solving failed with status {self.model.Status}")

        obj_value = self.model.ObjVal
        selected_routes = [self.routes[idx] for idx, var in enumerate(self.lambda_vars) if var.X > 0.5]
        return obj_value, selected_routes

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
