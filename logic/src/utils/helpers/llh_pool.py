"""
Low-Level Heuristic (LLH) pool for hyper-heuristics.
Provides basic operators for route improvement and perturbation,
supporting both single-route and multi-period collection plans.
"""

import copy
import random
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
from logic.src.utils.helpers.wrappers import (
    greedy_day_route,
    two_opt,
)

# --- Type Aliases ---
Plan = List[List[int]]
LLH = Callable[[Plan, ProblemContext, np.random.Generator], Plan]


# --- Intra-Route Operators (Single Route) ---


def h1_greedy_move(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """Insert a random profitable non-mandatory node."""
    nodes = set(range(1, len(problem.distance_matrix)))
    cand = list(nodes - set(route))
    if not cand:
        return route
    v = rng.choice(cand)
    if sum(problem.wastes.get(n, 0.0) for n in route) + problem.wastes.get(v, 0.0) <= problem.capacity:
        new_rt = route + [v]
        return two_opt(new_rt, problem.distance_matrix)
    return route


def h2_random_drop(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """Remove a random non-mandatory node."""
    cand = [v for v in route if v not in problem.mandatory]
    if not cand:
        return route
    v = rng.choice(cand)
    new_rt = [n for n in route if n != v]
    return two_opt(new_rt, problem.distance_matrix)


def h3_two_opt(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """Standard 2-opt search."""
    return two_opt(list(route), problem.distance_matrix)


def h4_swap_nodes(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """Swap two nodes in the route."""
    if len(route) < 2:
        return route
    new_rt = list(route)
    i, j = rng.choice(range(len(route)), size=2, replace=False)
    new_rt[i], new_rt[j] = new_rt[j], new_rt[i]
    return new_rt


def h5_relocate_node(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """Relocate a node to a different position."""
    if len(route) < 2:
        return route
    new_rt = list(route)
    v = new_rt.pop(rng.integers(len(new_rt)))
    new_rt.insert(rng.integers(len(new_rt) + 1), v)
    return new_rt


def h6_replace_node(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """Replace a node with a more profitable external node."""
    cand_ext = list(set(range(1, len(problem.distance_matrix))) - set(route))
    cand_in = [v for v in route if v not in problem.mandatory]
    if not cand_ext or not cand_in:
        return route
    v_in = rng.choice(cand_in)
    v_ext = rng.choice(cand_ext)
    # Check capacity
    if (
        sum(problem.wastes.get(n, 0.0) for n in route) - problem.wastes.get(v_in, 0.0) + problem.wastes.get(v_ext, 0.0)
        <= problem.capacity
    ):
        new_rt = [v_ext if n == v_in else n for n in route]
        return two_opt(new_rt, problem.distance_matrix)
    return route


def h7_greedy_multi(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """Try to insert multiple profitable nodes."""
    res = list(route)
    for _ in range(3):
        res = h1_greedy_move(problem, res, rng)
    return res


def h8_perturb_2opt(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """Random double bridge swap."""
    if len(route) < 4:
        return route
    # Simplified: just do 2 random swaps
    res = h4_swap_nodes(problem, route, rng)
    return h4_swap_nodes(problem, res, rng)


def h9_worst_removal(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """Remove node with highest distance contribution."""
    if len(route) < 2:
        return route
    path = [0] + route + [0]
    costs = []
    for i in range(len(route)):
        # cost added by node i: dist(prev, i) + dist(i, next) - dist(prev, next)
        c = (
            problem.distance_matrix[path[i]][path[i + 1]]
            + problem.distance_matrix[path[i + 1]][path[i + 2]]
            - problem.distance_matrix[path[i]][path[i + 2]]
        )
        # Weight by profit (low profit high cost is worst)
        p = problem.wastes.get(route[i], 1e-6)
        costs.append(c / p)

    idx = np.argmax(costs)
    if route[idx] in problem.mandatory:
        return route
    new_rt = [n for i, n in enumerate(route) if i != idx]
    return new_rt


def h10_greedy_shuffle(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """Shuffle route and 2-opt it."""
    new_rt = list(route)
    rng.shuffle(new_rt)
    return two_opt(new_rt, problem.distance_matrix)


# --- Multi-Period Operator Builders ---


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


# --- LLH Pools ---

INTRA_ROUTE_POOL = [
    h1_greedy_move,
    h2_random_drop,
    h3_two_opt,
    h4_swap_nodes,
    h5_relocate_node,
    h6_replace_node,
    h7_greedy_multi,
    h8_perturb_2opt,
    h9_worst_removal,
    h10_greedy_shuffle,
]

MULTI_PERIOD_LLH_POOL: List[Tuple[str, LLH]] = [
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

# Compatibility alias for the old single-route pool
LLH_POOL = INTRA_ROUTE_POOL


class LLHPool:
    """
    Low-Level Heuristic Pool for hyper-heuristics.
    Contains different perturbation/LS operators that the HH can select from.
    All LLHs take (plan, problem_context, rng) and return a new plan.
    """

    @staticmethod
    def llh_swap(plan: List[List[List[int]]], problem: ProblemContext, rng: random.Random) -> List[List[List[int]]]:
        """Swaps two random adjacent nodes in a random route."""
        new_plan = copy.deepcopy(plan)
        D = problem.horizon
        d = rng.randint(0, D - 1)
        rt = new_plan[d][0] if new_plan[d] else []
        if len(rt) >= 2:
            i = rng.randint(0, len(rt) - 2)
            rt[i], rt[i + 1] = rt[i + 1], rt[i]
            new_plan[d] = [rt]
        return new_plan

    @staticmethod
    def llh_drop_add(plan: List[List[List[int]]], problem: ProblemContext, rng: random.Random) -> List[List[List[int]]]:
        """Drops a node and adds a new one on a random day."""
        new_plan = copy.deepcopy(plan)
        D = problem.horizon
        d = rng.randint(0, D - 1)
        rt = new_plan[d][0] if new_plan[d] else []

        if rt:
            # drop non-mandatory
            cand_rem = [v for v in rt if v not in problem.mandatory]
            if cand_rem:
                rt.remove(rng.choice(cand_rem))

        # add valid
        cand_add = [v for v in range(1, len(problem.distance_matrix)) if v not in rt]
        if cand_add:
            v_add = rng.choice(cand_add)
            w_sum = sum(problem.wastes.get(n, 0.0) for n in rt)
            if w_sum + problem.wastes.get(v_add, 0.0) <= problem.capacity:
                rt.append(v_add)

        new_plan[d] = [rt]
        return new_plan

    @staticmethod
    def llh_2opt_all(plan: List[List[List[int]]], problem: ProblemContext, rng: random.Random) -> List[List[List[int]]]:
        """Applies 2-opt to all days."""
        new_plan = []
        for d in range(problem.horizon):
            rt = plan[d][0] if plan[d] else []
            new_plan.append([two_opt(list(rt), problem.distance_matrix)])
        return new_plan

    @staticmethod
    def llh_ruin_recreate_day(
        plan: List[List[List[int]]], problem: ProblemContext, rng: random.Random
    ) -> List[List[List[int]]]:
        """Completely reconstructs a random day."""
        new_plan = copy.deepcopy(plan)
        D = problem.horizon
        d = rng.randint(0, D - 1)

        # Advance context to day d
        cur_prob = problem
        for i in range(d):
            r = new_plan[i][0] if new_plan[i] else []
            cur_prob = cur_prob.advance(r)

        np_rng = np.random.default_rng(rng.randint(0, 2**31))
        rt = greedy_day_route(cur_prob, np_rng)
        new_plan[d] = [rt]
        return new_plan

    @classmethod
    def get_all(cls) -> List[Callable]:
        return [cls.llh_swap, cls.llh_drop_add, cls.llh_2opt_all, cls.llh_ruin_recreate_day]
