r"""
Branch-and-Cut Solver for VRPP using Gurobi.

This solver adapts the mathematical infrastructure of Fischetti et al. (1997)
for the Symmetric Generalized TSP to the Single-Vehicle VRPP. Nodes are treated
as Singleton Clusters, and SECs are implemented as Generalized Subtour Elimination
Constraints (GSECs) which reduce to Prize-Collecting SECs (PC-SECs).

References:
    Fischetti, M., Lodi, A., & Toth, P. (1997). "A Branch-and-Cut Algorithm for the Symmetric
    Generalized Traveling Salesman Problem". Operations Research, 45(2), 326-349.

    Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004). "A new branch-and-cut algorithm
    for the capacitated vehicle routing problem". Mathematical Programming, 100(2), 423-445.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.branch_and_cut.heuristics import (
    construct_initial_solution,
    construct_nn_solution,
    farthest_insertion,
)
from logic.src.policies.branch_and_cut.separation import (
    CapacityCut,
    PCSubtourEliminationCut,
    SeparationEngine,
)
from logic.src.policies.branch_and_cut.vrpp_model import VRPPModel

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp = None  # type: ignore[assignment]
    GRB = None  # type: ignore[assignment,misc]


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
        profit_aware_operators: bool = False,
        vrpp: bool = False,
        enable_fractional_capacity_cuts: bool = True,
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
            profit_aware_operators: Whether to use profit-aware heuristics.
            vrpp: Whether to use VRPP pool expansion in heuristics.
            enable_fractional_capacity_cuts: Enable exact fractional RCC separation.
                Recommended to set False for instances with n > 75 (see SeparationEngine docs).
        """
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is required for Branch-and-Cut solver")

        self.model = model
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.max_cuts_per_round = max_cuts_per_round
        self.use_heuristics = use_heuristics
        self.verbose = verbose
        self.profit_aware_operators = profit_aware_operators
        self.vrpp = vrpp

        # Separation engine with adaptive capacity cut toggling
        # Auto-disable for large instances (n > 75) to prevent O(V⁴) bottleneck
        if model.n_nodes > 75 and enable_fractional_capacity_cuts:
            if verbose:
                print(f"⚠ Large instance detected (n={model.n_nodes}). Disabling exact fractional capacity cuts.")
            enable_fractional_capacity_cuts = False

        self.separator = SeparationEngine(model, enable_fractional_capacity_cuts=enable_fractional_capacity_cuts)

        # Statistics
        self.stats = {
            "total_cuts": 0.0,
            "sec_cuts": 0.0,
            "capacity_cuts": 0.0,
            "nodes_explored": 0.0,
            "lp_iterations": 0.0,
        }

        # Gurobi model
        self.gurobi_model: Optional[gp.Model] = None
        self.x_vars: Dict[Tuple[int, int], gp.Var] = {}
        self.y_vars: Dict[int, gp.Var] = {}

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

        # Step 2: Root Node Strengthening via Lagrangian Relaxation
        # Achieves tight lower bounds and extracts Basic GSECs for injection
        violated_sets = self._pre_optimize_lagrangian()

        # Step 3: Inject Lagrangian GSECs directly into the root LP
        if violated_sets:
            if self.verbose:
                print(f"Injecting {len(violated_sets)} Lagrangian GSECs into root LP...")
            for s_set in violated_sets:
                # Add as Form (2.3) GSEC: sum_{e in delta(S)} x_e >= 2(yi + yj - 1)
                # For root strengthening, we can use the form that supports optional nodes.
                cut_edges = self.model.delta(s_set)
                edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

                # Pick any i in S and j not in S (typically ones with high y*)
                # Since we don't have y* yet, we'll use mandatory nodes or just any node
                node_i = next(iter(s_set))
                remaining_nodes = set(range(self.model.n_nodes)) - s_set
                node_j = next(iter(remaining_nodes))

                # Equation (2.3) form
                self.gurobi_model.addConstr(  # type: ignore[union-attr]
                    gp.quicksum(edge_vars) >= 2 * (self.y_vars.get(node_i, 1.0) + self.y_vars.get(node_j, 1.0) - 1.0),
                    name=f"lagrangian_gsec_{node_i}_{node_j}",
                )
            self.gurobi_model.update()  # type: ignore[union-attr]

        # Step 4: Get initial primal solution (heuristic)
        if self.use_heuristics:
            best_tour = []
            best_profit = -float("inf")

            # 1. Greedy Profit Initialization (Best for VRPP Knapsack constraints)
            tour_greedy, profit_greedy = construct_initial_solution(self.model)
            if profit_greedy > best_profit:
                best_profit, best_tour = profit_greedy, tour_greedy

            # 2. Nearest Neighbor Initialization (Good for tight spatial clustering)
            tour_nn, profit_nn = construct_nn_solution(self.model)
            if profit_nn > best_profit:
                best_profit, best_tour = profit_nn, tour_nn

            # 3. Farthest Insertion (Good for exploring the convex hull, as per Fischetti 1997)
            tour_farthest, profit_farthest = farthest_insertion(
                self.model,
                profit_aware_operators=self.profit_aware_operators,
                expand_pool=self.vrpp,
            )
            if profit_farthest > best_profit:
                best_profit, best_tour = profit_farthest, tour_farthest

            if best_tour:
                if self.verbose:
                    print(f"Warm start heuristic selected with profit: {best_profit:.2f}")
                self._set_start_solution(best_tour)

        # Step 3: Enable lazy constraint callback for cutting planes
        # CRITICAL: LazyConstraints=1 is MANDATORY for correctness.
        # Without MTZ variables, the base model CANNOT enforce subtours or capacity on its own.
        # The SeparationEngine dynamically detects and adds violated SEC/RCC cuts via callbacks.
        # If callbacks fail or are disabled, the solver will return invalid routes.
        assert self.gurobi_model is not None
        self.gurobi_model.Params.LazyConstraints = 1
        self.gurobi_model.Params.PreCrush = 1  # type: ignore[union-attr]
        self.gurobi_model.optimize(self._lazy_constraint_callback)  # type: ignore[union-attr]

        # Step 4: Extract solution
        routes, profit = self._extract_solution()

        self.stats["obj_value"] = profit
        assert self.gurobi_model is not None
        self.stats["solve_time"] = self.gurobi_model.Runtime
        self.stats["mip_gap"] = self.gurobi_model.MIPGap if self.gurobi_model.SolCount > 0 else 1.0
        self.stats["nodes_explored"] = int(self.gurobi_model.NodeCount)

        return routes, profit, self.stats  # type: ignore[return-value]

    def _build_initial_model(self):
        """Build the initial Gurobi model with basic constraints."""
        self.gurobi_model = gp.Model("VRPP_BranchAndCut")
        self.gurobi_model.Params.TimeLimit = self.time_limit
        self.gurobi_model.Params.MIPGap = self.mip_gap
        self.gurobi_model.Params.OutputFlag = 1 if self.verbose else 0

        # Decision variables
        # x[i,j]: Edge (i,j) is in the tour
        for i, j in self.model.edges:
            # Special case for depot edges: allow value 2 for [0, i, 0] trips
            if i == self.model.depot or j == self.model.depot:
                self.x_vars[(i, j)] = self.gurobi_model.addVar(lb=0, ub=2, vtype=GRB.INTEGER, name=f"x_{i}_{j}")
            else:
                self.x_vars[(i, j)] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # y[i]: Node i is visited
        for i in self.model.customers:
            self.y_vars[i] = self.gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

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

        # Depot degree constraint (allows up to K vehicles)
        depot_edges = [(u, v) for u, v in self.model.edges if self.model.depot in (u, v)]
        self.gurobi_model.addConstr(
            gp.quicksum(self.x_vars[e] for e in depot_edges) <= 2 * self.model.num_vehicles,
            name="depot_degree",
        )

        # 2. Mandatory node constraints
        for i in self.model.mandatory_nodes:
            if i in self.y_vars:
                self.gurobi_model.addConstr(self.y_vars[i] == 1, name=f"mandatory_{i}")

        # 3. Global capacity constraint (Knapsack)
        # sum of demands of visited nodes <= K * Q (aggregate capacity)
        self.gurobi_model.addConstr(
            gp.quicksum(self.model.get_node_demand(i) * self.y_vars[i] for i in self.model.customers)
            <= self.model.num_vehicles * self.model.capacity,
            name="global_capacity",
        )

        self.gurobi_model.update()

    def _lazy_constraint_callback(self, model, where):
        """
        Gurobi callback for Optimized Branch-and-Cut.

        Optimized Callback Structure:
            1. MIPSOL (Integer Solutions):
               - Only separates and adds Subtour Elimination Cuts (SECs).
               - These are mandatory for integer feasibility to prevent disconnected cycles.
               - Added via cbLazy().

            2. MIPNODE (Fractional LP Nodes):
               - Separates and adds both Capacity Cuts and SECs.
               - These act as User Cuts strengthening the LP relaxation bound.
               - Capacity Cuts are mathematically redundant for integer feasibility
                 due to the global knapsack constraint, but essential for a tight LP.
               - Added via cbCut().
        """
        if where == GRB.Callback.MIPSOL:
            # Integer solution: only SECs are mandatory for feasibility
            self._add_integer_cuts(model)

        elif where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            # Fractional LP node: add both SECs and Capacity Cuts as User Cuts
            self._add_fractional_cuts(model)

            # Custom Branching on Cuts (Section 6.2, Fischetti et al. 1997)
            # This strategy is superior to variable branching for large instances.
            # We search for a subset S where sum_{e in delta(S)} x_e is not an even integer.
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) % 2 == 0:  # Periodically check
                self._handle_custom_branching(model)

    def _handle_custom_branching(self, model):
        """
        Custom branching on cuts (Section 6.2, Fischetti et al. 1997).
        Identifies a subset S with non-even fractional cut value and branches.
        """
        # Get fractional solution for branching search
        x_var_list = [self.x_vars[(i, j)] for i, j in self.model.edges]
        x_vals = np.array(model.cbGetNodeRel(x_var_list))

        y_var_list = [self.y_vars[i] for i in self.model.customers]
        y_vals = np.array(model.cbGetNodeRel(y_var_list))

        # Separate to find candidate sets S
        cuts = self.separator.separate_fractional(
            x_vals, y_vals=y_vals, max_cuts=10, iteration=0, node_count=int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))
        )

        best_S = None
        best_frac = 0.0

        for cut in cuts:
            # We want sum_{e in delta(S)} x_e to be as close to 2k+1 as possible
            cut_edges = self.model.delta(cut.node_set)
            edge_vars_vals = [
                x_vals[self.model.edge_to_idx[tuple(sorted(e))]]
                for e in cut_edges
                if tuple(sorted(e)) in self.model.edge_to_idx
            ]
            val = sum(edge_vars_vals)

            # Check if fractional part is close to 1 (odd)
            frac = abs(val - 2 * round(val / 2))
            if frac > best_frac:
                best_frac = frac
                best_S = cut.node_set

        if best_S and best_frac > 0.1:
            cut_edges = self.model.delta(best_S)
            edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]

            val = sum(
                x_vals[self.model.edge_to_idx[tuple(sorted(e))]]
                for e in cut_edges
                if tuple(sorted(e)) in self.model.edge_to_idx
            )
            k = int(np.floor(val / 2.0))

            # Create two branches: sum x_e <= 2k and sum x_e >= 2k+2
            model.cbBranch(gp.quicksum(edge_vars) <= 2 * k)
            model.cbBranch(gp.quicksum(edge_vars) >= 2 * k + 2)

    def _add_integer_cuts(self, model):
        """
        Add violated cuts at integer solutions (MIPSOL callback).

        This method ensures correctness for the Natural Edge Formulation:
        1. Extracts the current integer solution (x_vals, y_vals) via cbGetSolution()
        2. Invokes the SeparationEngine to detect violated inequalities (fast heuristics)
        3. Adds violated cuts as lazy constraints via model.cbLazy()

        Correctness Contract (Fischetti et al. 1997):
            - If the solution contains a disconnected subtour, the SeparationEngine
              MUST detect it and return a SubtourEliminationCut.
            - If the solution violates vehicle capacity constraints for any node set,
              the SeparationEngine MUST detect it and return a CapacityCut.
            - If no violated cuts are found, the solution is guaranteed to be valid.

        Performance:
            - Uses fast connected-components check for SECs (NetworkX-based)
            - Heuristic capacity separation (demand-based clustering)
            - O(n²) complexity, suitable for integer callback
        """
        # Get current integer solution values using bulk API extraction
        # Performance Optimization: Bulk cbGetSolution() reduces Python-C API overhead
        x_var_list = [self.x_vars[(i, j)] for i, j in self.model.edges]
        y_var_list = [self.y_vars[i] for i in self.model.customers]

        x_vals = np.array(model.cbGetSolution(x_var_list))
        y_vals = np.array(model.cbGetSolution(y_var_list))

        # Separate violated inequalities (integer mode: SECs only)
        iteration = self.stats["lp_iterations"]
        cuts = self.separator.separate_integer(
            x_vals, y_vals=y_vals, max_cuts=self.max_cuts_per_round, iteration=iteration, sec_only=True
        )

        # Add cuts as lazy constraints
        for cut in cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                self._add_pcsec_lazy(model, cut)
                self.stats["sec_cuts"] += 1
            elif isinstance(cut, CapacityCut):
                self._add_capacity_cut_lazy(model, cut)
                self.stats["capacity_cuts"] += 1

            self.stats["total_cuts"] += 1

        self.stats["lp_iterations"] += 1

    def _add_fractional_cuts(self, model):
        """
        Add violated cuts at fractional LP relaxation nodes (MIPNODE callback).

        This method strengthens the LP relaxation during branch-and-bound:
        1. Extracts fractional solution values via cbGetNodeRel()
        2. Invokes exact separation (throttled max-flow) to find deep cuts
        3. Adds violated cuts via model.cbCut()

        Throttling Strategy (Lysgaard et al. 2004):
            - Exact max-flow separation is O(V⁴) and computationally expensive
            - Only run at root node (nodcnt == 0) or very shallow depths (nodcnt < 5)
            - Deeper in the tree, rely on integer cuts and Gurobi's internal cuts

        Performance:
            - Root node: Full exact separation (max-flow for SECs and RCCs)
            - Shallow nodes (depth < 5): Limited exact separation
            - Deep nodes: Skip fractional separation entirely
        """
        # Get current node count (depth indicator)
        node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)

        # Throttle expensive fractional separation based on tree depth
        # Strategy: Only run at root (node_count == 0) or very shallow depths
        if node_count > 4:
            # Deep in the tree: skip fractional separation to avoid bottleneck
            return

        # Get fractional solution values using bulk API extraction
        # Performance Optimization: Bulk cbGetNodeRel() is O(n) vs O(n²) for loop-based extraction
        # This reduces Python-C API overhead from ~1000 calls to 2 calls for n=50 instances
        x_var_list = [self.x_vars[(i, j)] for i, j in self.model.edges]
        y_var_list = [self.y_vars[i] for i in self.model.customers]

        x_vals = np.array(model.cbGetNodeRel(x_var_list))
        y_vals = np.array(model.cbGetNodeRel(y_var_list))

        # Separate violated inequalities (fractional mode: exact max-flow separation)
        iteration = self.stats["lp_iterations"]
        max_cuts = self.max_cuts_per_round // 2  # Limit cuts at fractional nodes

        cuts = self.separator.separate_fractional(
            x_vals, y_vals=y_vals, max_cuts=max_cuts, iteration=iteration, node_count=node_count
        )

        # Add cuts as user cuts (not lazy - for LP relaxation strengthening)
        for cut in cuts:
            if isinstance(cut, PCSubtourEliminationCut):
                self._add_pcsec_user(model, cut)
                self.stats["sec_cuts"] += 1
            elif isinstance(cut, CapacityCut):
                self._add_capacity_cut_user(model, cut)
                self.stats["capacity_cuts"] += 1

            self.stats["total_cuts"] += 1

    def _add_pcsec_lazy(self, model, cut: PCSubtourEliminationCut):
        """
        Add a prize-collecting subtour elimination cut (PC-SEC) as a lazy constraint.

        Lazy constraints are only checked for integer solutions and are essential
        for correctness in the Natural Edge Formulation.
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            # Fischetti et al. (1997) Facet Forms
            if cut.facet_form == "2.1":
                model.cbLazy(gp.quicksum(edge_vars) >= 2.0)
            elif cut.facet_form == "2.2":
                yi = self.y_vars[cut.node_i]
                model.cbLazy(gp.quicksum(edge_vars) >= 2 * yi)
            else:  # Form 2.3
                yi = self.y_vars[cut.node_i]
                yj = self.y_vars.get(cut.node_j, 1.0)
                model.cbLazy(gp.quicksum(edge_vars) >= 2 * (yi + yj - 1.0))

    def _add_capacity_cut_lazy(self, model, cut: CapacityCut):
        """
        Add a capacity cut as a lazy constraint (integer callback).

        Lazy constraints are only checked for integer solutions and are essential
        for correctness when capacity violations cannot be detected by base constraints.
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            model.cbLazy(gp.quicksum(edge_vars) >= cut.rhs)

    def _add_pcsec_user(self, model, cut: PCSubtourEliminationCut):
        """
        Add a prize-collecting subtour elimination cut (PC-SEC) as a user cut.

        User cuts strengthen the LP relaxation at fractional nodes but are not
        required for correctness (unlike lazy constraints).
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            # Fischetti et al. (1997) Facet Forms
            if cut.facet_form == "2.1":
                model.cbCut(gp.quicksum(edge_vars) >= 2.0)
            elif cut.facet_form == "2.2":
                yi = self.y_vars[cut.node_i]
                model.cbCut(gp.quicksum(edge_vars) >= 2 * yi)
            else:  # Form 2.3
                yi = self.y_vars[cut.node_i]
                yj = self.y_vars.get(cut.node_j, 1.0)
                model.cbCut(gp.quicksum(edge_vars) >= 2 * (yi + yj - 1.0))

    def _add_capacity_cut_user(self, model, cut: CapacityCut):
        """
        Add a capacity cut as a user cut (fractional callback).

        User cuts strengthen the LP relaxation at fractional nodes but are not
        required for correctness.
        """
        cut_edges = self.model.delta(cut.node_set)
        edge_vars = [self.x_vars[tuple(sorted(e))] for e in cut_edges if tuple(sorted(e)) in self.x_vars]  # type: ignore[index]

        if edge_vars:
            model.cbCut(gp.quicksum(edge_vars) >= cut.rhs)

    def _set_start_solution(self, tour: List[int]):
        """Provide a warm start solution to Gurobi."""
        # Set x variables based on tour
        for i, j in self.model.edges:
            self.x_vars[(i, j)].Start = 0.0

        for idx in range(len(tour) - 1):
            edge = tuple(sorted([tour[idx], tour[idx + 1]]))
            if edge in self.x_vars:
                # Increment start value to handle using same edge twice (e.g. [0, 1, 0])
                current_start = getattr(self.x_vars[edge], "Start", 0.0)  # type: ignore[index]
                if np.isnan(current_start):  # Gurobi might initialize with NaN
                    current_start = 0.0
                self.x_vars[edge].Start = current_start + 1.0  # type: ignore[index]

        # Set y variables based on visited nodes
        visited_nodes = set(tour) - {self.model.depot}
        for i in self.model.customers:
            self.y_vars[i].Start = 1.0 if i in visited_nodes else 0.0

    def _extract_solution(self) -> Tuple[List[List[int]], float]:
        """
        Extract the optimal routes from Gurobi solution.
        Handles multiple routes in the aggregate flow formulation.
        """
        assert self.gurobi_model is not None
        if self.gurobi_model.SolCount == 0:
            return [], 0.0

        # Extract active edges into a multiset-like list
        active_edges = []
        for (i, j), var in self.x_vars.items():
            val = round(var.X)
            for _ in range(int(val)):
                active_edges.append(list(sorted([i, j])))

        if not active_edges:
            return [], 0.0

        routes = []
        remaining_edges = active_edges

        # Multi-vehicle route extraction: trace all paths starting from depot
        while True:
            # Find an edge incident to depot
            start_edge = None
            for e in remaining_edges:
                if self.model.depot in e:
                    start_edge = e
                    break

            if start_edge is None:
                break

            # Start a new route
            route = [self.model.depot]
            current = start_edge[1] if start_edge[0] == self.model.depot else start_edge[0]
            remaining_edges.remove(start_edge)
            route.append(current)

            while current != self.model.depot:
                found_next = False
                for e in remaining_edges:
                    if current in e:
                        next_node = e[1] if e[0] == current else e[0]
                        remaining_edges.remove(e)
                        route.append(next_node)
                        current = next_node
                        found_next = True
                        break
                if not found_next:
                    # Should not happen if SECs are working correctly
                    break

            routes.append(route)

        # Compute total profit (obj value isMinimized travel_cost - waste_collected)
        profit = -self.gurobi_model.ObjVal

        return routes, profit

    def _pre_optimize_lagrangian(self) -> List[Set[int]]:
        """
        Root Node Strengthening via Lagrangian Relaxation.

        Refactored to extract violated Basic GSECs (1-tree cycles) for direct injection.
        """
        if self.verbose:
            print("Strengthening Root Node via Lagrangian Relaxation...")

        n = self.model.n_nodes
        # Initial multipliers
        lambda_mult = np.zeros(n)

        step_size = 0.5
        best_lower_bound = -float("inf")

        # Track unique sets S for injection
        unique_violated_sets: List[Set[int]] = []
        seen_sets = set()

        # Max 1000 iterations as per paper
        for iter_count in range(1000):
            # 1. Update edge weights with multipliers
            modified_costs = {}
            for i, j in self.model.edges:
                cost = self.model.get_edge_cost(i, j)
                modified_costs[(i, j)] = cost - lambda_mult[i] - lambda_mult[j]

            # 2. Solve Minimum Cost K-Tree (Generalized 1-tree for fleet K)
            # We extract all intermediate components (Basic PC-SECs) from Kruskal's
            k_tree_edges, lb, components = self._solve_k_tree(modified_costs)

            # Identify cycles in the K-tree
            cycle = self._find_k_tree_cycle(k_tree_edges)
            if cycle and self.model.depot not in cycle:
                unique_violated_sets.append(cycle)

            # Add all discovered components S
            for s_set in components:
                if self.model.depot not in s_set and len(s_set) >= 2:
                    s_tuple = tuple(sorted(list(s_set)))
                    if s_tuple not in seen_sets:
                        unique_violated_sets.append(s_set)
                        seen_sets.add(s_tuple)

            # Add constant part of Lagrangian: 2 * sum(lambda_i)
            current_lb = float(lb) + 2 * np.sum(lambda_mult)

            if current_lb > best_lower_bound:
                best_lower_bound = float(current_lb)

            # 3. Subgradient Update
            degrees = np.zeros(n)
            for i, j in k_tree_edges:
                degrees[i] += 1
                degrees[j] += 1

            # Subgradient for relaxed degree constraints: b - Ax
            # For customers: 2 - degree_i. For depot: 2K - degree_0.
            subgradient = 2.0 - degrees
            subgradient[0] = 2.0 * self.model.num_vehicles - degrees[0]

            if np.linalg.norm(subgradient) < 1e-6:
                break

            lambda_mult += (step_size / (iter_count + 1)) * subgradient

        # 4. Initialize variable set J using edges with small reduced costs
        for i, j in self.model.edges:
            cost = self.model.get_edge_cost(i, j)
            red_cost = cost - lambda_mult[i] - lambda_mult[j]
            if red_cost < 1e-4:
                self.x_vars[(i, j)].VarHintVal = 1.0

        if self.verbose:
            print(f"Lagrangian phase complete. Root LB: {best_lower_bound:.2f}")

        return unique_violated_sets

    def _find_k_tree_cycle(self, edges: List[Tuple[int, int]]) -> Optional[Set[int]]:
        """Find a cycle in a K-tree graph using DFS."""
        adj: Dict[int, List[int]] = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        visited = set()
        path = []

        def dfs(u, p):
            visited.add(u)
            path.append(u)
            for v in adj.get(u, []):
                if v == p:
                    continue
                if v in visited:
                    # Cycle found
                    cycle_start_idx = path.index(v)
                    return set(path[cycle_start_idx:])
                res = dfs(v, u)
                if res:
                    return res
            path.pop()
            return None

        for node in range(self.model.n_nodes):
            if node not in visited:
                res = dfs(node, -1)
                if res:
                    return res
        return None

    def _solve_k_tree(self, costs: Dict[Tuple[int, int], float]) -> Tuple[List[Tuple[int, int]], float, List[Set[int]]]:
        """
        Solve Minimum Cost K-Tree subproblem (Fischetti et al. 1997).
        Generalized from 1-tree to handle fleet size K.

        Returns:
            Tuple of (edges, cost, all_components_from_kruskal).
        """
        n = self.model.n_nodes
        # MST on nodes 1...n-1
        customers = self.model.customers
        edges_to_mst = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                u, v = customers[i], customers[j]
                edges_to_mst.append((u, v, costs[(u, v)]))

        edges_to_mst.sort(key=lambda x: x[2])

        # Kruskal's with component tracking
        parent = list(range(n))
        node_to_comp = {i: {i} for i in range(n)}
        history_components = []

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        mst_edges = []
        mst_cost = 0.0
        for u, v, cost in edges_to_mst:
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                mst_edges.append((u, v))
                mst_cost += cost

                # Merge components and store history
                new_comp = node_to_comp[root_u] | node_to_comp[root_v]
                history_components.append(new_comp)

                parent[root_u] = root_v
                node_to_comp[root_v] = new_comp

        # Select up to 2K depot edges based on reduced costs (Fischetti 1997)
        # Select all edges with negative reduced costs, but at least 2 and at most 2K.
        depot_edges = []
        for j in self.model.customers:
            depot_edges.append((0, j, costs[(0, j)]))
        depot_edges.sort(key=lambda x: x[2])

        k = self.model.num_vehicles
        selected_depot_edges: List[Tuple[int, int, float]] = []
        for u, v, cost in depot_edges:
            if len(selected_depot_edges) < 2 * k:
                # Always take at least 2 edges to ensure connectedness
                if len(selected_depot_edges) < 2 or cost < 0:
                    selected_depot_edges.append((u, v, cost))
            else:
                break

        k_tree_edges = mst_edges + [(e[0], e[1]) for e in selected_depot_edges]
        k_tree_cost = mst_cost + sum(e[2] for e in selected_depot_edges)

        return k_tree_edges, k_tree_cost, history_components
