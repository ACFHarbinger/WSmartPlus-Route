import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.route_construction.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.utils import (
    greedy_day_route,
    route_cost,
    route_profit,
    two_opt,
)


@RouteConstructorRegistry.register("mp_ils")
class ILSPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Iterated Local Search (ILS) metaheuristic.
    1. Initial solution (greedy + 2-opt)
    2. Local search (steepest descent)
    3. Perturbation (destroy and recreate)
    4. Acceptance criterion
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        cfg = config or {}
        self.max_iter = getattr(cfg, "max_iter", 100)
        self.perturb_size = getattr(cfg, "perturb_size", 3)
        self.seed = getattr(cfg, "seed", 42)
        self.rng = random.Random(self.seed)

    def _evaluate(self, plan: List[List[List[int]]], problem: ProblemContext) -> float:
        tot = 0.0
        cur_prob = problem
        for d in range(problem.horizon):
            rt = plan[d][0] if plan[d] else []
            tot += route_profit(rt, cur_prob)
            cur_prob = cur_prob.advance(rt)
        return tot

    def _local_search(self, plan: List[List[List[int]]], problem: ProblemContext) -> List[List[List[int]]]:
        # Apply 2-opt to each day
        new_plan = []
        cur_prob = problem
        for d in range(problem.horizon):
            rt = plan[d][0] if plan[d] else []
            opt_rt = two_opt(list(rt), cur_prob.distance_matrix)
            new_plan.append([opt_rt])
            cur_prob = cur_prob.advance(opt_rt)
        return new_plan

    def _perturb(self, plan: List[List[List[int]]], problem: ProblemContext) -> List[List[List[int]]]:
        # Destroys `perturb_size` nodes and tries to re-insert them
        # (or just simple random drop for a VRPP where we can drop)
        new_plan = copy.deepcopy(plan)
        cur_prob = problem

        for d in range(problem.horizon):
            rt = new_plan[d][0] if new_plan[d] else []
            if not rt:
                cur_prob = cur_prob.advance(rt)
                continue

            n_remove = min(self.perturb_size, len(rt))
            rem = set(self.rng.sample(rt, n_remove))
            rt = [v for v in rt if v not in rem]

            # recreate phase: optionally try greedy insertion of non-mandatory
            # (In VRPP, we don't have to re-insert if it's not mandatory)
            mand_rem = rem.intersection(set(cur_prob.mandatory))
            for m in mand_rem:
                rt.append(m)  # force append mandatory at end, we'll LS it later

            new_plan[d] = [rt]
            cur_prob = cur_prob.advance(rt)

        return new_plan

    def _run_multi_period_solver(
        self, problem: ProblemContext, multi_day_ctx: Optional[MultiDayContext]
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        D = problem.horizon
        np_rng = np.random.default_rng(self.seed)

        # Initial
        cur_plan = []
        cur_prob = problem
        for _d in range(D):
            rt = greedy_day_route(cur_prob, np_rng)
            cur_plan.append([rt])
            cur_prob = cur_prob.advance(rt)

        cur_plan = self._local_search(cur_plan, problem)
        best_plan = copy.deepcopy(cur_plan)
        best_prof = self._evaluate(best_plan, problem)
        cur_prof = best_prof

        for _ in range(self.max_iter):
            perturbed = self._perturb(cur_plan, problem)
            improved = self._local_search(perturbed, problem)

            imp_prof = self._evaluate(improved, problem)

            # Acceptance (simple greedy)
            if imp_prof > cur_prof:
                cur_plan = copy.deepcopy(improved)
                cur_prof = imp_prof

                if imp_prof > best_prof:
                    best_plan = copy.deepcopy(improved)
                    best_prof = imp_prof

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_single_route(
            route=today_route,
            profit=route_profit(today_route, problem),
            cost=route_cost(today_route, problem),
            metadata={"ils_iters": self.max_iter},
        )
        return sol, best_plan, {"ils_iters": self.max_iter}
