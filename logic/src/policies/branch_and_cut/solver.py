"""
Branch-and-Cut Solver for VRPP using Gurobi.

Implements the enumerative algorithm described in Section 5 of Fischetti et al. (1997),
adapted for the Vehicle Routing Problem with Profits.

Reference:
    Fischetti, M., Lodi, A., & Toth, P. (1997). "A Branch-and-Cut Algorithm for the Symmetric
    Generalized Traveling Salesman Problem". Operations Research, 45(2), 326-349.
    Padberg, M., & Rinaldi, G. (1991). "A Branch-and-cut Algorithm for the Resolution of
    Large-scale Symmetric Traveling Salesman Problems". SIAM Review, 33(1), 60-100.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp = None
    GRB = None

from logic.src.policies.branch_and_cut.heuristics import construct_initial_solution
from logic.src.policies.branch_and_cut.separation import (
    CapacityCut,
    SeparationEngine,
    SubtourEliminationCut,
)
from logic.src.policies.branch_and_cut.vrpp_model import VRPPModel


class BranchAndCutSolver:
    """
    Branch-and-Cut solver for VRPP.

    Follows the cutting plane algorithm described in Section 5.1 of Fischetti et al. (1997):
    1. Solve LP relaxation
    2. Separate violated inequalities
    3. Add cuts and resolve
    4. Branch if fractional
    5. Use primal heuristics for upper bounds
    """

    def __init__(
        self,
        model: VRPPModel,
        time_limit: float = 300.0,
        mip_gap: float = 0.01,
        max_cuts_per_round: int = 50,
        use_heuristics: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize Branch-and-Cut solver.

        Args:
            model: VRPPModel instance defining the problem.
            time_limit: Maximum solving time in seconds.
            mip_gap: Relative MIP gap tolerance.
            max_cuts_per_round: Maximum cuts to add per separation round.
            use_heuristics: Whether to use primal heuristics.
            verbose: Print detailed logging.
        """
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is required for Branch-and-Cut solver")

        self.model = model
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.max_cuts_per_round = max_cuts_per_round
        self.use_heuristics = use_heuristics
        self.verbose = verbose

        # Separation engine
        self.separator = SeparationEngine(model)

        # Statistics
        self.stats = {
            "total_cuts": 0,
            "sec_cuts": 0,
            "capacity_cuts": 0,
            "nodes_explored": 0,
            "lp_iterations": 0,
        }

        # Gurobi model
        self.gurobi_model: Optional[gp.Model] = None
        self.x_vars: Dict[Tuple[int, int], gp.Var] = {}
        self.y_vars: Dict[int, gp.Var] = {}
        self.u_vars: Dict[int, gp.Var] = {}

    def solve(self) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        Solve the VRPP instance.

        Returns:
            Tuple of (tour, profit, statistics).
        """
        if self.verbose:
            print("=" * 60)
            print("Branch-and-Cut Solver for VRPP")
            print("=" * 60)
            print(f"Nodes: {self.model.n_nodes}")
            print(f"Capacity: {self.model.capacity}")
            print(f"Mandatory nodes: {len(self.model.mandatory_nodes)}")
            print("=" * 60)

        # Step 1: Build initial Gurobi model
        self._build_initial_model()

        # Step 2: Get initial primal solution (heuristic)
        if self.use_heuristics:
            initial_tour, initial_profit = construct_initial_solution(self.model)
            if initial_tour:
                self._set_start_solution(initial_tour)
                if self.verbose:
                    print(f"Heuristic solution profit: {initial_profit:.2f}")

        # Step 3: Enable lazy constraint callback for cutting planes
        self.gurobi_model.Params.LazyConstraints = 1
        self.gurobi_model.Params.PreCrush = 1  # Enable presolve
        self.gurobi_model.optimize(self._lazy_constraint_callback)

        # Step 4: Extract solution
        tour, profit = self._extract_solution()

        self.stats["obj_value"] = profit
        self.stats["solve_time"] = self.gurobi_model.Runtime
        self.stats["mip_gap"] = self.gurobi_model.MIPGap if self.gurobi_model.SolCount > 0 else 1.0
        self.stats["nodes_explored"] = int(self.gurobi_model.NodeCount)

        if self.verbose:
            print("=" * 60)
            print("Solution Statistics:")
            print(f"  Profit: {profit:.2f}")
            print(f"  Tour length: {len(tour)}")
            print(f"  Nodes visited: {len(set(tour)) - 1}")  # Exclude depot
            print(f"  Total cuts added: {self.stats['total_cuts']}")
            print(f"    - SEC cuts: {self.stats['sec_cuts']}")
            print(f"    - Capacity cuts: {self.stats['capacity_cuts']}")
            print(f"  Nodes explored: {self.stats['nodes_explored']}")
            print(f"  MIP gap: {self.stats['mip_gap']:.2%}")
            print(f"  Runtime: {self.stats['solve_time']:.2f}s")
            print("=" * 60)

        return tour, profit, self.stats

    def _build_initial_model(self):
        """Build the initial Gurobi model with basic constraints."""
        self.gurobi_model = gp.Model("VRPP_BranchAndCut")
        self.gurobi_model.Params.TimeLimit = self.time_limit
        self.gurobi_model.Params.MIPGap = self.mip_gap
        self.gurobi_model.Params.OutputFlag = 1 if self.verbose else 0

        # Decision variables
        # x[i,j]: Edge (i,j) is in the tour
        for i, j in self.model.edges:
            self.x_vars[(i, j)] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # y[i]: Node i is visited
        for i in self.model.customers:
            self.y_vars[i] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

        # u[i]: Cumulative load after visiting i (MTZ variables)
        for i in self.model.customers:
            self.u_vars[i] = self.gurobi_model.addVar(
                lb=0.0, ub=self.model.capacity, vtype=GRB.CONTINUOUS, name=f"u_{i}"
            )

        # Objective: Maximize profit = waste collected - travel cost
        # Gurobi minimizes by default, so negate
        waste_collected = gp.quicksum(self.model.get_node_profit(i) * self.y_vars[i] for i in self.model.customers)

        travel_cost = gp.quicksum(self.model.get_edge_cost(i, j) * self.x_vars[(i, j)] for i, j in self.model.edges)

        self.gurobi_model.setObjective(travel_cost - waste_collected, GRB.MINIMIZE)

        # Constraints
        # 1. Degree constraints: sum of edges incident to i equals 2*y[i]
        for i in self.model.customers:
            incident_edges = [(u, v) for u, v in self.model.edges if i in (u, v)]
            self.gurobi_model.addConstr(
                gp.quicksum(self.x_vars[e] for e in incident_edges) == 2 * self.y_vars[i],
                name=f"degree_{i}",
            )

        # Depot degree constraint (always exactly 2 edges if any visit)
        depot_edges = [(u, v) for u, v in self.model.edges if self.model.depot in (u, v)]
        self.gurobi_model.addConstr(
            gp.quicksum(self.x_vars[e] for e in depot_edges) == 2,
            name="depot_degree",
        )

        # 2. Mandatory node constraints
        for i in self.model.mandatory_nodes:
            if i in self.y_vars:
                self.gurobi_model.addConstr(self.y_vars[i] == 1, name=f"mandatory_{i}")

        # 3. MTZ capacity constraints (basic subtour elimination)
        for i in self.model.customers:
            for j in self.model.customers:
                if i != j:
                    edge_ij = tuple(sorted([i, j]))
                    if edge_ij in self.x_vars:
                        demand_j = self.model.get_node_demand(j)
                        self.gurobi_model.addConstr(
                            self.u_vars[j]
                            >= self.u_vars[i] + demand_j - self.model.capacity * (1 - self.x_vars[edge_ij]),
                            name=f"mtz_{i}_{j}",
                        )

        self.gurobi_model.update()

    def _lazy_constraint_callback(self, model, where):
        """
        Gurobi callback for lazy constraint generation.

        This is called at integer solutions to add violated cutting planes.
        """
        if where == GRB.Callback.MIPSOL:
            # We found an integer solution - check for violated cuts
            self._add_lazy_cuts(model)

    def _add_lazy_cuts(self, model):
        """Add violated cuts at integer solutions."""
        # Get current solution values
        x_vals = np.array([model.cbGetSolution(self.x_vars[(i, j)]) for i, j in self.model.edges])

        # Separate violated inequalities
        iteration = self.stats["lp_iterations"]
        cuts = self.separator.separate(x_vals, max_cuts=self.max_cuts_per_round, iteration=iteration)

        # Add cuts as lazy constraints
        for cut in cuts:
            if isinstance(cut, SubtourEliminationCut):
                self._add_subtour_cut(model, cut)
                self.stats["sec_cuts"] += 1
            elif isinstance(cut, CapacityCut):
                self._add_capacity_cut(model, cut)
                self.stats["capacity_cuts"] += 1

            self.stats["total_cuts"] += 1

        self.stats["lp_iterations"] += 1

    def _add_subtour_cut(self, model, cut: SubtourEliminationCut):
        """Add a subtour elimination cut to the model."""
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]

        if edge_vars:
            model.cbLazy(gp.quicksum(edge_vars) >= cut.rhs)

    def _add_capacity_cut(self, model, cut: CapacityCut):
        """Add a capacity inequality cut to the model."""
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]

        if edge_vars:
            model.cbLazy(gp.quicksum(edge_vars) >= cut.rhs)

    def _set_start_solution(self, tour: List[int]):
        """Provide a warm start solution to Gurobi."""
        # Set x variables based on tour
        for i, j in self.model.edges:
            self.x_vars[(i, j)].Start = 0.0

        for idx in range(len(tour) - 1):
            edge = tuple(sorted([tour[idx], tour[idx + 1]]))
            if edge in self.x_vars:
                self.x_vars[edge].Start = 1.0

        # Set y variables based on visited nodes
        visited_nodes = set(tour) - {self.model.depot}
        for i in self.model.customers:
            self.y_vars[i].Start = 1.0 if i in visited_nodes else 0.0

    def _extract_solution(self) -> Tuple[List[int], float]:
        """Extract the optimal tour from Gurobi solution."""
        if self.gurobi_model.SolCount == 0:
            return [], 0.0

        # Extract active edges
        active_edges = []
        for (i, j), var in self.x_vars.items():
            if var.X > 0.5:
                active_edges.append((i, j))

        if not active_edges:
            return [], 0.0

        # Build adjacency list
        adj: Dict[int, List[int]] = {i: [] for i in range(self.model.n_nodes)}
        for i, j in active_edges:
            adj[i].append(j)
            adj[j].append(i)

        # Extract tour starting from depot
        tour = [self.model.depot]
        current = self.model.depot

        visited_edges = set()
        while True:
            next_nodes = [
                n for n in adj[current] if (current, n) not in visited_edges and (n, current) not in visited_edges
            ]

            if not next_nodes:
                break

            next_node = next_nodes[0]
            visited_edges.add((current, next_node))
            visited_edges.add((next_node, current))
            tour.append(next_node)
            current = next_node

            if current == self.model.depot:
                break

        # Compute profit
        profit = -self.gurobi_model.ObjVal  # Negate because we minimized

        return tour, profit
