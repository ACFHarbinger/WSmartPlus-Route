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


@RouteConstructorRegistry.register("gp_mp_hh")
class GPHeuristicPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Genetic Programming Hyper-Heuristic (GP-HH).
    Evolves a sequence (program) of LLHs to apply.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        cfg = config or {}
        self.pop_size = getattr(cfg, "pop_size", 10)
        self.gens = getattr(cfg, "gens", 20)
        self.prog_len = getattr(cfg, "prog_len", 5)
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

        # Initial base plan
        cur_plan = []
        cur_prob = problem
        for _d in range(problem.horizon):
            rt = greedy_day_route(cur_prob, np_rng)
            cur_plan.append([rt])
            cur_prob = cur_prob.advance(rt)

        # Population of sequences of LLH indices
        pop = [[self.rng.randint(0, len(self.llhs) - 1) for _ in range(self.prog_len)] for _ in range(self.pop_size)]

        best_plan = None
        best_prof = -float("inf")

        for _ in range(self.gens):
            fitnesses = []
            plans = []

            for j in range(self.pop_size):
                prog = pop[j]
                p = copy.deepcopy(cur_plan)

                # Execute program
                for idx in prog:
                    cmd = self.llhs[idx]
                    p = cmd(p, problem, self.rng)

                prof = self._evaluate(p, problem)
                fitnesses.append(prof)
                plans.append(p)

                if prof > best_prof:
                    best_prof = prof
                    best_plan = copy.deepcopy(p)

            # Selection and crossover (simple tournament + 1pt Xover)
            new_pop = []
            for _ in range(self.pop_size):
                # Tournament
                t1, t2 = self.rng.sample(range(self.pop_size), 2)
                p1 = pop[t1] if fitnesses[t1] > fitnesses[t2] else pop[t2]

                t1, t2 = self.rng.sample(range(self.pop_size), 2)
                p2 = pop[t1] if fitnesses[t1] > fitnesses[t2] else pop[t2]

                xpt = self.rng.randint(1, self.prog_len - 1)
                child = p1[:xpt] + p2[xpt:]

                # Mutate
                if self.rng.random() < 0.1:
                    mpt = self.rng.randint(0, self.prog_len - 1)
                    child[mpt] = self.rng.randint(0, len(self.llhs) - 1)

                new_pop.append(child)

            pop = new_pop

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_single_route(
            route=today_route,
            profit=route_profit(today_route, problem),
            cost=route_cost(today_route, problem),
            metadata={"gphh_gens": self.gens},
        )
        return sol, best_plan, {"gphh_gens": self.gens}
