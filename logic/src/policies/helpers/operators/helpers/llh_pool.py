# logic/src/policies/helpers/operators/helpers/llh_pool.py

from typing import Callable, List, Tuple

import numpy as np

from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.policies.helpers.operators.intra_route_local_search import (
    or_opt_route,
    three_opt_route,
    two_opt_route,
)
from logic.src.policies.helpers.operators.perturbation_shaking import (
    cross_day_move,
    cross_day_swap,
    day_merge_split,
    day_shuffle,
    double_bridge,
)

Plan = List[List[int]]
LLH = Callable[[Plan, ProblemContext, np.random.Generator], Plan]


def _make_or_opt(chain_len: int) -> LLH:
    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        d = int(rng.integers(0, len(plan)))
        new_route = or_opt_route(plan[d], problem.distance_matrix, chain_lengths=(chain_len,))
        new_plan = list(plan)
        new_plan[d] = new_route
        return new_plan

    return _op


def _make_two_opt() -> LLH:
    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        d = int(rng.integers(0, len(plan)))
        new_plan = list(plan)
        new_plan[d] = two_opt_route(plan[d], problem.distance_matrix)
        return new_plan

    return _op


def _make_three_opt() -> LLH:
    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        d = int(rng.integers(0, len(plan)))
        new_plan = list(plan)
        new_plan[d] = three_opt_route(plan[d], problem.distance_matrix)
        return new_plan

    return _op


def _make_cross_day_move() -> LLH:
    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        all_pairs = [(n, d) for d, route in enumerate(plan) for n in route]
        if not all_pairs:
            return plan
        bin_id, src_day = all_pairs[int(rng.integers(0, len(all_pairs)))]
        tgt_day = int(rng.integers(0, len(plan)))
        new_plan, _ = cross_day_move(plan, problem, bin_id, src_day, tgt_day)
        return new_plan

    return _op


def _make_cross_day_swap() -> LLH:
    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        all_pairs = [(n, d) for d, route in enumerate(plan) for n in route]
        if len(all_pairs) < 2:
            return plan
        idx = rng.choice(len(all_pairs), size=2, replace=False)
        (bin_i, day_i), (bin_j, day_j) = all_pairs[idx[0]], all_pairs[idx[1]]
        new_plan, _ = cross_day_swap(plan, problem, bin_i, day_i, bin_j, day_j)
        return new_plan

    return _op


def _make_day_merge_split() -> LLH:
    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        new_plan, _ = day_merge_split(plan, problem, rng)
        return new_plan

    return _op


def _make_double_bridge() -> LLH:
    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        d = int(rng.integers(0, len(plan)))
        new_plan = list(plan)
        double_bridge(new_plan, d, rng)
        return new_plan

    return _op


def _make_day_shuffle() -> LLH:
    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        new_plan, _ = day_shuffle(plan, problem, rng, k=3)
        return new_plan

    return _op


LLH_POOL: List[Tuple[str, LLH]] = [
    ("or_opt_1", _make_or_opt(1)),
    ("or_opt_2", _make_or_opt(2)),
    ("or_opt_3", _make_or_opt(3)),
    ("two_opt", _make_two_opt()),
    ("three_opt", _make_three_opt()),
    ("cross_day_move", _make_cross_day_move()),
    ("cross_day_swap", _make_cross_day_swap()),
    ("day_merge_split", _make_day_merge_split()),
    ("double_bridge", _make_double_bridge()),
    ("day_shuffle", _make_day_shuffle()),
]
