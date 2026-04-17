import math
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp

from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry


@RouteConstructorRegistry.register("rfo")
class RelaxFixOptimizePolicy(BaseMultiPeriodRoutingPolicy):
    """
    Relax-and-Fix Meta-heuristic.
    Given a horizon D, iterate over time windows.
    In each step, variables before the window are FIXED,
    variables in the window are INTEGER,
    and variables after the window are RELAXED (Continuous).
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        cfg = config or {}
        self.window_size = getattr(cfg, "window_size", 3)
        self.step_size = getattr(cfg, "step_size", 2)
        self.mip_time = getattr(cfg, "mip_time", 60.0)
        self.mip_gap = getattr(cfg, "mip_gap", 0.01)

    def _setup_multi_day_model(
        self, problem: ProblemContext, int_days: set, relaxed_days: set, fixed_plan: Dict[int, List[int]]
    ):
        model = gp.Model("RFO")
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.mip_time)
        model.setParam("MIPGap", self.mip_gap)
        model.Params.LazyConstraints = 1

        N = len(problem.distance_matrix)
        x = {}  # x[d, i, j]
        y = {}  # y[d, i]

        # Simplified Multi-day formulation: independent routing per day with shared capacity limit
        # This acts as an approximation; real cascade requires complex modeling of fill dynamics
        # For this template, we construct a block structured model.
        for d in range(problem.horizon):
            var_type = gp.GRB.BINARY if d in int_days else gp.GRB.CONTINUOUS
            for i in range(1, N):
                y[d, i] = model.addVar(vtype=var_type, lb=0, ub=1, name=f"y_{d}_{i}")
                x[d, 0, i] = model.addVar(vtype=var_type, lb=0, ub=1, name=f"x_{d}_0_{i}")
                x[d, i, 0] = model.addVar(vtype=var_type, lb=0, ub=1, name=f"x_{d}_{i}_0")
                for j in range(1, N):
                    if i != j:
                        x[d, i, j] = model.addVar(vtype=var_type, lb=0, ub=1, name=f"x_{d}_{i}_{j}")

        # Build constraints for each day
        for d in range(problem.horizon):
            # Degree constraints
            for i in range(1, N):
                model.addConstr(gp.quicksum(x[d, i, j] for j in range(N) if j != i) == y[d, i])
                model.addConstr(gp.quicksum(x[d, j, i] for j in range(N) if j != i) == y[d, i])

            # Single vehicle degree at depot
            model.addConstr(gp.quicksum(x[d, 0, j] for j in range(1, N)) <= 1)
            model.addConstr(gp.quicksum(x[d, j, 0] for j in range(1, N)) <= 1)

            # Capacity (using day 0 fixed weights approximation for simplicity in MIP)
            model.addConstr(gp.quicksum(problem.wastes.get(i, 0.0) * y[d, i] for i in range(1, N)) <= problem.capacity)

            # If day is already fixed, fix the variables
            if d in fixed_plan:
                route = fixed_plan[d]
                r_set = set(route)
                for i in range(1, N):
                    y[d, i].LB = 1.0 if i in r_set else 0.0
                    y[d, i].UB = 1.0 if i in r_set else 0.0

                path = [0] + route + [0]
                edges = {(path[k], path[k + 1]) for k in range(len(path) - 1)}
                for i in range(N):
                    for j in range(N):
                        if i != j and (d, i, j) in x:
                            val = 1.0 if (i, j) in edges else 0.0
                            x[d, i, j].LB = val
                            x[d, i, j].UB = val

        # Objective
        obj = gp.quicksum(
            problem.revenue_per_kg * problem.wastes.get(i, 0.0) * y[d, i]
            - problem.cost_per_km * x[d, i, j] * problem.distance_matrix[i, j]
            for d in range(problem.horizon)
            for i in range(N)
            for j in range(N)
            if i != j and (d, i, j) in x
        )
        model.setObjective(obj, gp.GRB.MAXIMIZE)

        return model, x, y

    def _run_multi_period_solver(
        self, problem: ProblemContext, multi_day_ctx: Optional[MultiDayContext]
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        D = problem.horizon
        fixed_plan: Dict[int, List[int]] = {}

        for w_start in range(0, D, self.step_size):
            w_end = min(D, w_start + self.window_size)
            int_days = set(range(w_start, w_end))
            relaxed_days = set(range(w_end, D))

            model, x, y = self._setup_multi_day_model(problem, int_days, relaxed_days, fixed_plan)

            # Subtour separation needs to handle multi-day structure nicely.
            # We'll pass a custom lambda or just skip lazy for day-by-day continuous loops during fix block
            model.Params.LazyConstraints = 0  # simplifying lazy subtour for full multiday approximation
            model.optimize()

            # Extract routes for the integer portion and save
            if model.SolCount > 0:
                for d in range(w_start, w_start + self.step_size):
                    if d >= D:
                        break
                    # reconstruct flat day tour
                    route = []
                    curr = 0
                    N = len(problem.distance_matrix)
                    for _ in range(N):
                        nxt = -1
                        for j in range(N):
                            if j != curr and x[d, curr, j].X > 0.5:
                                nxt = j
                                break
                        if nxt == -1 or nxt == 0:
                            break
                        route.append(nxt)
                        curr = nxt
                    fixed_plan[d] = route
            else:
                for d in range(w_start, w_start + self.step_size):
                    if d < D:
                        fixed_plan[d] = []

        # Construct final plan List[List[List[int]]]
        full_plan = []
        for d in range(D):
            full_plan.append([fixed_plan.get(d, [])])

        today_route = full_plan[0][0] if full_plan[0] else []

        # calculate route profit/cost dynamically
        from logic.src.policies.route_construction.matheuristics.utils import route_cost, route_profit

        cost = route_cost(today_route, problem)
        profit = route_profit(today_route, problem)

        sol = SolutionContext.from_single_route(route=today_route, profit=profit, cost=cost, metadata={})
        return sol, full_plan, {"rfo_windows_solved": math.ceil(D / self.step_size)}
