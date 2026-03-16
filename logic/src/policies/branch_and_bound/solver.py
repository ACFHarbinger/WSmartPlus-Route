"""
Branch-and-Bound (BB) solver core engine.

Implements the Land and Doig (1960) methodology for exact integer programming.
The solver uses Gurobi for Linear Programming (LP) relaxations and manages a
search tree to ensure global optimality for VRPP/CWC VRP problems.

Reference:
    Land, A. H., & Doig, A. (1960). "An automatic method for solving discrete
    programming problems". Econometrica, 28(3), 497-520.
"""

import heapq
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .node import Node


class BBSolver:
    """
    Exact Branch-and-Bound engine for Vehicle Routing Problems.

    This engine is designed to solve the Profit-Maximizing or Waste-Collecting
    Vehicle Routing Problem with high precision, utilizing automated
    variable selection and node pruning based on deterministic bounds.

    The solver maintains an 'incumbent' solution (the best integer solution
    found so far) and stops when the search tree is exhausted or the
    specified MIP gap is reached.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        values: Dict[str, Any],
        must_go_indices: Optional[Set[int]] = None,
        env: Optional[gp.Env] = None,
        recorder: Optional[PolicyStateRecorder] = None,
    ):
        """
        Initialize the Branch-and-Bound solver.

        Args:
            dist_matrix: Symmetric matrix representing edge costs.
            wastes: Mapping customer IDs to their respective fill levels.
            capacity: Physical payload limit for the routing model.
            R: Revenue coefficient per unit of collected waste.
            C: Cost coefficient per unit of distance traveled.
            values: Merged configuration dictionary from hydra and adapter.
            must_go_indices: Set of customer nodes that MUST be included in routes.
            env: Optional Gurobi environment for resource management.
            recorder: Optional telemetry recorder for state tracking.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.values = values
        self.must_go_indices = must_go_indices or set()
        self.env = env
        self.recorder = recorder

        self.num_nodes = len(dist_matrix)
        self.nodes_range = range(self.num_nodes)
        self.customers = list(range(1, self.num_nodes))

        self.time_limit = values.get("time_limit", 60.0)
        self.mip_gap = values.get("mip_gap", 0.01)
        self.branching_strategy = values.get("branching_strategy", "most_fractional")

        self.incumbent_obj = float("inf")
        self.incumbent_routes: List[List[int]] = []
        self.start_time = 0.0

    def _setup_relaxation_model(self, node: Node) -> Tuple[gp.Model, Dict, Dict, Dict]:
        """
        Construct and solve the Linear Programming relaxation for a search node.

        Variable fixing is enforced via lower and upper bounds in the LP model,
        reflecting the integrity constraints of the current search branch.

        Args:
            node: The current search node containing fixed variable states.

        Returns:
            A tuple of (Solved Model, X-variables, Y-variables, U-variables).
        """
        model = gp.Model("BB_Relaxation", env=self.env) if self.env else gp.Model("BB_Relaxation")
        model.setParam("OutputFlag", 0)  # Silence Gurobi output for sub-solves
        model.setParam("Threads", 1)  # Single-threaded sub-solves for better control

        # --- Decision Variables (Continuous [0, 1] for Relaxation) ---

        # x[i,j]: Edge usage probability
        x = {}
        for i in self.nodes_range:
            for j in self.nodes_range:
                if i != j:
                    lb, ub = 0.0, 1.0
                    if (i, j) in node.fixed_x:
                        lb = ub = float(node.fixed_x[(i, j)])
                    x[i, j] = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

        # y[i]: Node visit probability (VRPP/CWC)
        y = {}
        for i in self.customers:
            lb, ub = 0.0, 1.0
            if i in node.fixed_y:
                lb = ub = float(node.fixed_y[i])
            elif i in self.must_go_indices:
                lb = 1.0
            y[i] = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"y_{i}")

        # u[i]: Miller-Tucker-Zemlin (MTZ) cumulative load variables
        u = {}
        for i in self.customers:
            u[i] = model.addVar(lb=self.wastes.get(i, 0), ub=self.capacity, vtype=GRB.CONTINUOUS, name=f"u_{i}")

        # --- Objective: Minimize (Travel Cost + Opportunity Costs) ---
        travel_cost = gp.quicksum(self.dist_matrix[i][j] * self.C * x[i, j] for (i, j) in x)
        revenue_penalty = gp.quicksum((1 - y[i]) * self.wastes.get(i, 0) * self.R for i in self.customers)
        model.setObjective(travel_cost + revenue_penalty, GRB.MINIMIZE)

        # --- Base Constraints (Flow and Integrity) ---
        for i in self.customers:
            model.addConstr(gp.quicksum(x[i, j] for j in self.nodes_range if (i, j) in x) == y[i], name=f"out_{i}")
            model.addConstr(gp.quicksum(x[j, i] for j in self.nodes_range if (j, i) in x) == y[i], name=f"in_{i}")

        # --- MTZ Subtour Elimination ---
        for i in self.customers:
            for j in self.customers:
                if i != j:
                    d_j = self.wastes.get(j, 0)
                    model.addConstr(u[j] >= u[i] + d_j - self.capacity * (1 - x[i, j]), name=f"mtz_{i}_{j}")

        model.optimize()
        return model, x, y, u

    def _is_integer(self, val: float, tol: float = 1e-6) -> bool:
        """Heuristic check for integer feasibility of a continuous variable."""
        return abs(val - round(val)) < tol

    def _get_branching_variable(self, x: Dict, y: Dict) -> Optional[Tuple[str, Any]]:
        """
        Identify the most suitable fractional variable to branch upon.

        Supported strategies:
        - 'most_fractional': Maximize the uncertainty (closest to 0.5).
        - 'least_fractional': Minimize the uncertainty (closest to integer bounds).

        Args:
            x: Map of edge variables and their relaxed values.
            y: Map of node variables and their relaxed values.

        Returns:
            A tuple identifying the variable ('x'|'y', Index) or None if feasible.
        """
        fractional_vars = []

        for i_j, var in x.items():
            if not self._is_integer(var.X):
                fractional_vars.append(("x", i_j, var.X))

        for i, var in y.items():
            if not self._is_integer(var.X):
                fractional_vars.append(("y", i, var.X))

        if not fractional_vars:
            return None

        if self.branching_strategy == "most_fractional":
            return sorted(fractional_vars, key=lambda v: abs(v[2] - 0.5))[0][:2]

        return sorted(fractional_vars, key=lambda v: min(v[2], 1 - v[2]))[0][:2]

    def solve(self) -> Tuple[List[List[int]], float]:
        """
        Perform the Branch-and-Bound search to find the optimal integer routes.

        Implements the main traversal loop: popping nodes from priority queue,
        solving relaxations, updating the incumbent, and branching.

        Returns:
            A tuple of (Optimal Integer Routes, Total Objective Value).
        """
        self.start_time = time.time()

        # Initialize root subproblem
        root_model, root_x, root_y, root_u = self._setup_relaxation_model(Node(bound=0.0))
        if root_model.status != GRB.OPTIMAL:
            return [], 0.0

        # Priority Queue ensures we always explore the node with the best bound
        pq = [Node(bound=root_model.ObjVal)]

        nodes_explored = 0
        while pq and (time.time() - self.start_time < self.time_limit):
            node = heapq.heappop(pq)
            nodes_explored += 1

            # Pruning by bound: Stop if this branch cannot beat the best integer solution
            if node.bound >= self.incumbent_obj * (1 - self.mip_gap):
                continue

            # Solve the subproblem at this node
            model, x, y, u = self._setup_relaxation_model(node)
            if model.status != GRB.OPTIMAL:
                continue

            current_obj = model.ObjVal
            if current_obj >= self.incumbent_obj * (1 - self.mip_gap):
                continue

            # Check for integer feasibility
            branch_var = self._get_branching_variable(x, y)

            if branch_var is None:
                # Integer Feasible Node Found: Update Incumbent
                self.incumbent_obj = current_obj
                self.incumbent_routes = self._extract_routes(x)
            else:
                # Variable remains fractional: Branch and continue search
                var_type, var_idx = branch_var

                # Zero-Branch (Force variable to 0)
                node_0 = Node(
                    bound=current_obj, fixed_x=node.fixed_x.copy(), fixed_y=node.fixed_y.copy(), depth=node.depth + 1
                )
                if var_type == "x":
                    node_0.fixed_x[var_idx] = 0
                else:
                    node_0.fixed_y[var_idx] = 0
                heapq.heappush(pq, node_0)

                # One-Branch (Force variable to 1)
                node_1 = Node(
                    bound=current_obj, fixed_x=node.fixed_x.copy(), fixed_y=node.fixed_y.copy(), depth=node.depth + 1
                )
                if var_type == "x":
                    node_1.fixed_x[var_idx] = 1
                else:
                    node_1.fixed_y[var_idx] = 1
                heapq.heappush(pq, node_1)

        # Telemetry updates
        if self.recorder:
            self.recorder.record(
                engine="bb_land_doig",
                nodes_explored=nodes_explored,
                obj_val=self.incumbent_obj,
                time=time.time() - self.start_time,
            )

        return self.incumbent_routes, self.incumbent_obj

    def _extract_routes(self, x_vars: Dict) -> List[List[int]]:
        """
        Convert sparse binary edge variables into ordered route sequences.

        Traverses the solution adjacency graph starting from the depot (0).
        """
        edges = [e for e, v in x_vars.items() if v.X > 0.5]
        adj: Dict[int, List[int]] = {i: [] for i in self.nodes_range}
        for i, j in edges:
            adj[i].append(j)

        routes = []
        for start_node in adj[0]:
            route = [start_node]
            curr = start_node
            while curr != 0 and adj[curr]:
                next_node = adj[curr][0]
                if next_node != 0:
                    route.append(next_node)
                curr = next_node
            routes.append(route)
        return routes


def run_bb(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    must_go_indices: Optional[Set[int]] = None,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[List[int]], float]:
    """
    Dispatcher entry point for the Branch-and-Bound solver.

    Constructs the solver instance and executes the combinatorial search.
    """
    solver = BBSolver(dist_matrix, wastes, capacity, R, C, values, must_go_indices, env, recorder)
    return solver.solve()
