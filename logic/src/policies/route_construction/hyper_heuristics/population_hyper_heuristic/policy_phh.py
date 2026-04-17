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


@RouteConstructorRegistry.register("phh")
class PopulationHHPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Population-based Hyper-Heuristic (PHH).
    Applies selection hyper-heuristic independently to each individual in a population.
    Individuals periodically exchange LLH sequences or histories.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        cfg = config or {}
        self.pop_size = getattr(cfg, "pop_size", 10)
        self.gens = getattr(cfg, "gens", 20)
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

        pop = []
        best_plan = None
        best_prof = -float("inf")

        # Init pop
        for _ in range(self.pop_size):
            cur_plan = []
            cur_prob = problem
            for _d in range(problem.horizon):
                rt = greedy_day_route(cur_prob, np_rng)
                cur_plan.append([rt])
                cur_prob = cur_prob.advance(rt)
            pop.append(cur_plan)

        for _ in range(self.gens):
            for i in range(self.pop_size):
                p = pop[i]

                # Apply random LLH
                llh = self.rng.choice(self.llhs)
                p_new = llh(p, problem, self.rng)

                prof_old = self._evaluate(p, problem)
                prof_new = self._evaluate(p_new, problem)

                if prof_new > prof_old:
                    pop[i] = p_new
                    if prof_new > best_prof:
                        best_prof = prof_new
                        best_plan = copy.deepcopy(p_new)

            # Simplistic cross-pollination (tournament replacement)
            fits = [self._evaluate(x, problem) for x in pop]
            worst_idx = np.argmin(fits)
            best_idx = np.argmax(fits)
            pop[worst_idx] = copy.deepcopy(pop[best_idx])

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_single_route(
            route=today_route,
            profit=route_profit(today_route, problem),
            cost=route_cost(today_route, problem),
            metadata={"phh_gens": self.gens},
        )
        return sol, best_plan, {"phh_gens": self.gens}
