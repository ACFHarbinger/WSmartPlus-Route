"""
Master Problem for Branch-and-Price VRPP.

Implements the set partitioning formulation where each column represents a feasible route.
The master problem selects routes to cover all mandatory nodes while maximizing profit.

Based on Section 3.2 of Barnhart et al. (1998).
"""

from typing import Dict, List, Optional, Set, Tuple

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
    ):
        """
        Initialize a route.

        Args:
            nodes: Sequence of nodes visited (excluding depot returns)
            cost: Total distance cost of route
            revenue: Total revenue from collected waste
            load: Total waste collected
            node_coverage: Set of nodes covered by this route
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
    Set Partitioning Master Problem for VRPP with Column Generation.

    Formulation:
        max  Σ_k profit_k * λ_k
        s.t. Σ_k a_{ik} * λ_k = 1    ∀i ∈ mandatory nodes  (partitioning)
             Σ_k a_{ik} * λ_k ≤ 1    ∀i ∈ optional nodes   (covering)
             λ_k ∈ {0,1}             ∀k                    (route selection)

    where:
        - λ_k: binary variable indicating if route k is selected
        - a_{ik}: 1 if node i is in route k, 0 otherwise
        - profit_k: profit of route k (revenue - cost)
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
    ):
        """
        Initialize the master problem.

        Args:
            n_nodes: Number of customer nodes (excluding depot)
            mandatory_nodes: Set of node indices that must be visited
            cost_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 is depot
            wastes: Dictionary mapping node ID to waste volume
            capacity: Vehicle capacity
            revenue_per_kg: Revenue per unit of waste collected
            cost_per_km: Cost per unit of distance traveled
        """
        self.n_nodes = n_nodes
        self.mandatory_nodes = mandatory_nodes
        self.optional_nodes = set(range(1, n_nodes + 1)) - mandatory_nodes
        self.cost_matrix = cost_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = revenue_per_kg
        self.C = cost_per_km
        self.depot = 0

        # Column pool
        self.routes: List[Route] = []

        # Gurobi model
        self.model: Optional[gp.Model] = None
        self.lambda_vars: List[gp.Var] = []

        # Dual values (for pricing)
        self.dual_node_coverage: Dict[int, float] = {}
        self.dual_convexity: float = 0.0

    def add_route(self, route: Route) -> None:
        """Add a route (column) to the master problem."""
        self.routes.append(route)

        if self.model is not None:
            # Add variable to existing model
            self._add_column_to_model(route)

    def _add_column_to_model(self, route: Route) -> None:
        """Add a column to the Gurobi model."""
        if self.model is None:
            return

        # Create variable
        var = self.model.addVar(
            obj=route.profit,
            vtype=GRB.BINARY if not self.model.Params.RelaxIntegral else GRB.CONTINUOUS,
            name=f"route_{len(self.lambda_vars)}",
        )

        # Add to coverage constraints
        for node in route.node_coverage:
            constr_name = f"coverage_{node}"
            constr = self.model.getConstrByName(constr_name)
            if constr is not None:
                self.model.chgCoeff(constr, var, 1.0)

        self.lambda_vars.append(var)
        self.model.update()

    def build_model(self, initial_routes: Optional[List[Route]] = None) -> None:
        """
        Build the Gurobi model for the master problem.

        Args:
            initial_routes: Initial set of routes to include
        """
        self.model = gp.Model("VRPP_Master")
        self.model.Params.OutputFlag = 0
        self.lambda_vars = []

        if initial_routes:
            self.routes = initial_routes

        # Create variables for existing routes
        for idx, route in enumerate(self.routes):
            var = self.model.addVar(
                obj=route.profit,
                vtype=GRB.BINARY,
                name=f"route_{idx}",
            )
            self.lambda_vars.append(var)

        # Coverage constraints for mandatory nodes (equality)
        for node in self.mandatory_nodes:
            lhs = gp.LinExpr()
            for idx, route in enumerate(self.routes):
                if node in route.node_coverage:
                    lhs += self.lambda_vars[idx]

            self.model.addConstr(
                lhs == 1,
                name=f"coverage_{node}",
            )

        # Coverage constraints for optional nodes (inequality)
        for node in self.optional_nodes:
            lhs = gp.LinExpr()
            for idx, route in enumerate(self.routes):
                if node in route.node_coverage:
                    lhs += self.lambda_vars[idx]

            self.model.addConstr(
                lhs <= 1,
                name=f"coverage_{node}",
            )

        self.model.ModelSense = GRB.MAXIMIZE
        self.model.update()

    def solve_lp_relaxation(self) -> Tuple[float, Dict[int, float]]:
        """
        Solve the LP relaxation of the master problem.

        Returns:
            Tuple of (objective_value, route_values)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Relax integrality
        for var in self.lambda_vars:
            var.vtype = GRB.CONTINUOUS

        self.model.update()
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"LP relaxation failed with status {self.model.Status}")

        # Extract solution
        obj_value = self.model.ObjVal
        route_values = {idx: var.X for idx, var in enumerate(self.lambda_vars)}

        # Extract dual values
        self.dual_node_coverage = {}
        for node in range(1, self.n_nodes + 1):
            constr = self.model.getConstrByName(f"coverage_{node}")
            if constr is not None:
                self.dual_node_coverage[node] = constr.Pi

        return obj_value, route_values

    def solve_ip(self) -> Tuple[float, List[Route]]:
        """
        Solve the integer program.

        Returns:
            Tuple of (objective_value, selected_routes)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Set integrality
        for var in self.lambda_vars:
            var.vtype = GRB.BINARY

        self.model.update()
        self.model.optimize()

        if self.model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            raise RuntimeError(f"IP solving failed with status {self.model.Status}")

        # Extract solution
        obj_value = self.model.ObjVal
        selected_routes = []

        for idx, var in enumerate(self.lambda_vars):
            if var.X > 0.5:  # Binary variable
                selected_routes.append(self.routes[idx])

        return obj_value, selected_routes

    def get_reduced_cost_coefficients(self) -> Dict[int, float]:
        """
        Get the coefficients for computing reduced costs in the pricing subproblem.

        For a route covering nodes i, the reduced cost is:
            reduced_cost = profit - Σ_i dual_i

        Returns:
            Dictionary mapping node ID to dual value
        """
        return self.dual_node_coverage.copy()
