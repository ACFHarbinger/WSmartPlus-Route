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
from logic.src.policies.route_construction.matheuristics.utils import (
    greedy_day_route,
    route_cost,
    route_profit,
)


@RouteConstructorRegistry.register("shh")
class SelectionHHPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Selection Hyper-Heuristic (SHH).
    Controls a set of Low-Level Heuristics (LLHs).
    Uses a reinforcement learning-like mechanism (UCB or simple probability updates)
    to select the next LLH to apply, then accepts or rejects using Late Acceptance.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        cfg = config or {}
        self.iters = getattr(cfg, "iters", 200)
        self.history_len = getattr(cfg, "history_len", 10)
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

        cur_plan = []
        cur_prob = problem
        for _d in range(problem.horizon):
            rt = greedy_day_route(cur_prob, np_rng)
            cur_plan.append([rt])
            cur_prob = cur_prob.advance(rt)

        best_plan = copy.deepcopy(cur_plan)
        best_prof = self._evaluate(best_plan, problem)

        # Late Acceptance history
        la_history = [best_prof] * self.history_len

        # LLH weights
        weights = [1.0] * len(self.llhs)

        cur_prof = best_prof

        for k in range(self.iters):
            # Selection
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            idx = self.rng.choices(range(len(self.llhs)), weights=probs, k=1)[0]
            llh = self.llhs[idx]

            # Apply
            new_plan = llh(cur_plan, problem, self.rng)
            new_prof = self._evaluate(new_plan, problem)

            # Acceptance (Late Acceptance Hill Climbing)
            v = k % self.history_len
            if new_prof >= la_history[v] or new_prof >= cur_prof:
                cur_plan = copy.deepcopy(new_plan)
                cur_prof = new_prof

                # Reward LLH
                weights[idx] += 1.0

                if new_prof > best_prof:
                    best_prof = new_prof
                    best_plan = copy.deepcopy(new_plan)
            else:
                # Penalize slightly
                weights[idx] = max(0.1, weights[idx] - 0.1)

            la_history[v] = cur_prof

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_single_route(
            route=today_route,
            profit=route_profit(today_route, problem),
            cost=route_cost(today_route, problem),
            metadata={"shh_iters": self.iters},
        )
        return sol, best_plan, {"shh_iters": self.iters}
