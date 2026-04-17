import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.helpers.operators.helpers.llh_pool import LLHPool
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.utils import greedy_day_route, route_cost, route_profit


@RouteConstructorRegistry.register("mhh")
class MatheuristicHHPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Matheuristic Hyper-Heuristic (MHH).
    Combines LLHs for diversification and exact MIP (Relax and Fix) for intensification.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        cfg = config or {}
        self.iters = getattr(cfg, "iters", 10)
        self.seed = getattr(cfg, "seed", 42)
        self.rng = random.Random(self.seed)
        self.llhs = LLHPool.get_all()

    def _evaluate(self, plan: List[List[List[int]]], problem: ProblemContext) -> float:
        tot = 0.0
        cur_prob = problem
        for d in range(problem.horizon):
            rt = plan[d][0] if plan[d] else []
            tot += route_profit(rt, cur_prob)
            cur_prob = cur_prob.advance(rt)
        return tot

    def _run_multi_period_solver(
        self, problem: ProblemContext, multi_day_ctx: Optional[MultiDayContext]
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        np_rng = np.random.default_rng(self.seed)

        # Initial plan
        cur_plan = []
        cur_prob = problem
        for _d in range(problem.horizon):
            rt = greedy_day_route(cur_prob, np_rng)
            cur_plan.append([rt])
            cur_prob = cur_prob.advance(rt)

        best_plan = copy.deepcopy(cur_plan)
        best_prof = self._evaluate(best_plan, problem)

        try:
            from logic.src.policies.route_construction.matheuristics.relax_fix_optimize.policy_rfo import (
                RelaxFixOptimizePolicy,
            )

            rfo = RelaxFixOptimizePolicy(self.config)
        except Exception:
            rfo = None

        for _ in range(self.iters):
            # 1. Diversify with random LLH
            idx = self.rng.randint(0, len(self.llhs) - 1)
            p_div = self.llhs[idx](cur_plan, problem, self.rng)

            # 2. Intensify using exact solver
            if rfo:
                # We could feed p_div as a warm start or call it in RFO, but for now we just
                # call RFO to solve based on problem state (RFO does not take warm start plan in its interface natively unless extended).
                # We simply run RFO directly and use it if it's better.
                rfo_sol, rfo_plan, _ = rfo._run_multi_period_solver(problem, multi_day_ctx)
                p_int = rfo_plan
            else:
                p_int = copy.deepcopy(p_div)

            prof_int = self._evaluate(p_int, problem)
            if prof_int > best_prof:
                best_prof = prof_int
                best_plan = copy.deepcopy(p_int)

            # Random restart or keep going from int
            if self.rng.random() < 0.2:
                cur_plan = copy.deepcopy(p_int)

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_single_route(
            route=today_route,
            profit=route_profit(today_route, problem),
            cost=route_cost(today_route, problem),
            metadata={"mhh_iters": self.iters},
        )
        return sol, best_plan, {"mhh_iters": self.iters}
