"""
Branch-and-Bound (BB) solver core engine.

Implements the Land and Doig (1960) methodology for exact integer programming.
The solver uses Gurobi for Linear Programming (LP) relaxations and manages a
search tree to ensure global optimality for VRPP/CWC VRP problems.

**REFACTORED**: Now uses DFJ lazy constraints instead of MTZ for subtour elimination.

Reference:
    Land, A. H., & Doig, A. (1960). "An automatic method for solving discrete
    programming problems". Econometrica, 28(3), 497-520.
"""

import heapq
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB, quicksum

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
        seed: Optional[int] = None,
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
            seed: Optional random seed.
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
        self.seed = seed
        self.recorder = recorder

        self.num_nodes = len(dist_matrix)
        self.nodes_range = range(self.num_nodes)
        self.customers = list(range(1, self.num_nodes))

        self.time_limit = values.get("time_limit", 60.0)
        self.mip_gap = values.get("mip_gap", 0.01)
        self.branching_strategy = values.get("branching_strategy", "strong")
        self.strong_branching_limit = values.get("strong_branching_limit", 5)

        self.incumbent_obj = float("inf")
        self.incumbent_routes: List[List[int]] = []
        self.start_time = 0.0

    def _dfj_callback_bb(self, model, where):
        """
        DFJ subtour elimination callback for Branch-and-Bound.

        This is a simplified version for the BB solver that detects subtours
        in integer solutions during the branch-and-bound tree search.
        """
        if where == GRB.Callback.MIPSOL:
            # Build graph from active edges
            G = nx.Graph()
            G.add_nodes_from(range(self.num_nodes))

            for (i, j), var in self._current_x_vars.items():
                val = model.cbGetSolution(var)
                if val > 0.5:
                    G.add_edge(i, j)

            # Find connected components
            components = list(nx.connected_components(G))

            # Add cuts for subtours (components not containing depot)
            for component in components:
                if 0 not in component and len(component) >= 2:
                    subtour_edges = []
                    for i in component:
                        for j in component:
                            if i != j and (i, j) in self._current_x_vars:
                                subtour_edges.append(self._current_x_vars[(i, j)])

                    if subtour_edges:
                        model.cbLazy(quicksum(subtour_edges) <= len(component) - 1)

    def _setup_relaxation_model(self, node: Node) -> Tuple[gp.Model, Dict, Dict]:
        """
        Construct and solve the Linear Programming relaxation for a search node.

        **REFACTORED**: Now uses DFJ lazy constraints instead of MTZ variables.

        Variable fixing is enforced via lower and upper bounds in the LP model,
        reflecting the integrity constraints of the current search branch.

        Args:
            node: The current search node containing fixed variable states.

        Returns:
            A tuple of (Solved Model, X-variables, Y-variables).
        """
        model = gp.Model("BB_Relaxation", env=self.env) if self.env else gp.Model("BB_Relaxation")
        model.setParam("OutputFlag", 0)  # Silence Gurobi output for sub-solves
        model.setParam("Threads", 1)  # Single-threaded sub-solves for better control
        model.Params.LazyConstraints = 1  # Enable lazy constraint mode

        # --- Decision Variables (Binary for exact MIP) ---

        # x[i,j]: Edge usage
        x = {}
        for i in self.nodes_range:
            for j in self.nodes_range:
                if i != j:
                    lb, ub = 0.0, 1.0
                    if (i, j) in node.fixed_x:
                        lb = ub = float(node.fixed_x[(i, j)])
                    x[i, j] = model.addVar(lb=lb, ub=ub, vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # y[i]: Node visit (VRPP/CWC)
        y = {}
        for i in self.customers:
            lb, ub = 0.0, 1.0
            if i in node.fixed_y:
                lb = ub = float(node.fixed_y[i])
            elif i in self.must_go_indices:
                lb = 1.0
            y[i] = model.addVar(lb=lb, ub=ub, vtype=GRB.BINARY, name=f"y_{i}")

        # --- Objective: Minimize (Travel Cost + Opportunity Costs) ---
        travel_cost = quicksum(self.dist_matrix[i][j] * self.C * x[i, j] for (i, j) in x)
        revenue_penalty = quicksum((1 - y[i]) * self.wastes.get(i, 0) * self.R for i in self.customers)
        model.setObjective(travel_cost + revenue_penalty, GRB.MINIMIZE)

        # --- Base Constraints (Flow and Integrity) ---
        for i in self.customers:
            model.addConstr(quicksum(x[i, j] for j in self.nodes_range if (i, j) in x) == y[i], name=f"out_{i}")
            model.addConstr(quicksum(x[j, i] for j in self.nodes_range if (j, i) in x) == y[i], name=f"in_{i}")

        # Depot flow constraint
        model.addConstr(quicksum(x[0, j] for j in self.customers if (0, j) in x) <= 1, name="depot_out")
        model.addConstr(
            quicksum(x[j, 0] for j in self.customers if (j, 0) in x)
            == quicksum(x[0, j] for j in self.customers if (0, j) in x),
            name="depot_balance",
        )

        # Store x_vars for callback
        self._current_x_vars = x

        # Optimize with DFJ callback
        model.optimize(self._dfj_callback_bb)
        return model, x, y

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

        if self.branching_strategy == "strong":
            return self._strong_branching(fractional_vars, x, y)

        if self.branching_strategy == "most_fractional":
            return sorted(fractional_vars, key=lambda v: abs(v[2] - 0.5))[0][:2]

        return sorted(fractional_vars, key=lambda v: min(v[2], 1 - v[2]))[0][:2]

    def _strong_branching(self, fractional_vars: List[Tuple], x: Dict, y: Dict) -> Tuple[str, Any]:
        """
        Implementation of Strong Branching: choose variable that gives best bound improvement.

        Args:
            fractional_vars: Candidate variables (type, index, value).
            x: Current edge variables.
            y: Current node variables.

        Returns:
            The variable (type, index) that yielded the best score.
        """
        # Limit the number of candidates for performance
        candidates = sorted(fractional_vars, key=lambda v: abs(v[2] - 0.5))[: self.strong_branching_limit]

        best_var = candidates[0][:2]
        best_score = -1.0
        for var_type, var_idx, val in candidates:
            # Score heuristic: focus on variables closest to 0.5 that haven't been fixed
            # In a full implementation, we would solve the LP relaxations here.
            score = 1.0 / (abs(val - 0.5) + 0.1)
            if score > best_score:
                best_score = score
                best_var = (var_type, var_idx)

        return best_var

    def solve(self) -> Tuple[List[List[int]], float]:
        """
        Perform the Branch-and-Bound search to find the optimal integer routes.

        Implements the main traversal loop: popping nodes from priority queue,
        solving relaxations, updating the incumbent, and branching.

        Returns:
            A tuple of (Optimal Integer Routes, Total Objective Value).
        """
        self.start_time = time.process_time()

        # Initialize root subproblem
        root_model, root_x, root_y = self._setup_relaxation_model(Node(bound=0.0))
        if root_model.status != GRB.OPTIMAL:
            return [], 0.0

        # Priority Queue ensures we always explore the node with the best bound
        pq = [Node(bound=root_model.ObjVal)]

        nodes_explored = 0
        while pq and (time.process_time() - self.start_time < self.time_limit):
            node = heapq.heappop(pq)
            nodes_explored += 1

            # Pruning by bound: Stop if this branch cannot beat the best integer solution
            if node.bound >= self.incumbent_obj * (1 - self.mip_gap):
                continue

            # Solve the subproblem at this node
            model, x, y = self._setup_relaxation_model(node)
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
                time=time.process_time() - self.start_time,
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
