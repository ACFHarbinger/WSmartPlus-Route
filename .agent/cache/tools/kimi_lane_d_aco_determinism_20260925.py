import sys
sys.path.insert(0, "/tmp/kimi-wsr")
import numpy as np
import logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco as hyper_aco_mod
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco import HyperHeuristicACO
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.params import HyperACOParams
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
import random

rng = np.random.default_rng(0)
n = 9
coords = rng.uniform(0, 10, size=(n, 2))
dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
wastes = {i: float(rng.uniform(5, 40)) for i in range(1, n)}
capacity, R, C = 60.0, 1.0, 1.0
mandatory = [1, 2, 3]
init = build_greedy_routes(dist_matrix=dist, wastes=wastes, capacity=capacity, R=R, C=C,
                           mandatory_nodes=mandatory, rng=random.Random(42))
p = HyperACOParams(n_ants=6, max_iterations=8, time_limit=30, seed=42, profit_aware_operators=True)

# A) same seed, 6 repetitions -> how many distinct route sets?
outs = []
for _ in range(6):
    r = HyperHeuristicACO(dist, wastes, capacity, R, C, p, init, mandatory).solve()
    outs.append((tuple(tuple(x) for x in r[0]), round(r[1], 6)))
print("=== [e] reproducibility with fixed seed=42, 6 runs ===")
print("distinct (routes,profit) outcomes:", len(set(outs)), "of", len(outs))
profits = [o[1] for o in outs]
print("profits:", profits)

# B) patch perf_counter to a deterministic counter -> runs must match
calls = {"k": 0}
def fake_perf():
    calls["k"] += 1
    return calls["k"] * 1e-6
real_perf = hyper_aco_mod.time.perf_counter
hyper_aco_mod.time.perf_counter = fake_perf
try:
    a = HyperHeuristicACO(dist, wastes, capacity, R, C, p, init, mandatory).solve()
    b = HyperHeuristicACO(dist, wastes, capacity, R, C, p, init, mandatory).solve()
finally:
    hyper_aco_mod.time.perf_counter = real_perf
print("with deterministic clock, run A == run B:", a == b)
print("=> timing-dependence hypothesis confirmed" if (len(set(outs)) > 1 and a == b) else "=> hypothesis NOT confirmed")

# C) strategic oscillation: can the returned best_routes violate capacity?
rng2 = np.random.default_rng(7)
n2 = 13
coords2 = rng2.uniform(0, 10, size=(n2, 2))
dist2 = np.sqrt(((coords2[:, None, :] - coords2[None, :, :]) ** 2).sum(-1))
wastes2 = {i: float(rng2.uniform(15, 45)) for i in range(1, n2)}
cap2 = 50.0
init2 = build_greedy_routes(dist_matrix=dist2, wastes=wastes2, capacity=cap2, R=1.0, C=1.0,
                            mandatory_nodes=list(range(1, n2)), rng=random.Random(42))
def feasible(routes):
    return all(sum(wastes2.get(x, 0) for x in r) <= cap2 + 1e-9 for r in routes)
print("\n=== [b] strategic oscillation / infeasible best ===")
print("initial greedy feasible:", feasible(init2))
p_osc = HyperACOParams(n_ants=6, max_iterations=60, time_limit=25, seed=42,
                       profit_aware_operators=True, stagnation_limit=5, elitism_ratio=1.0)
s = HyperHeuristicACO(dist2, wastes2, cap2, 1.0, 1.0, p_osc, init2, list(range(1, n2)))
best = s.solve()
print("returned best_routes feasible:", feasible(best[0]), "| pv at end:", s.pv, "| initial_pv:", s.initial_pv)
for r in best[0]:
    print("  route", r, "load", round(sum(wastes2[x] for x in r), 1), "> cap", cap2, ":", sum(wastes2[x] for x in r) > cap2)
