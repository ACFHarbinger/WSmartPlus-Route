import sys, traceback
sys.path.insert(0, "/tmp/kimi-wsr")
import numpy as np

from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.policy_aco_hh import HyperACOPolicy
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco import HyperHeuristicACO
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators import (
    HYPER_OPERATORS, OPERATOR_NAMES, HyperOperatorContext, apply_string_removal,
)
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.params import HyperACOParams

print("=== [a] operator vertices vs yaml 'operators' list ===")
print("HYPER_OPERATORS (%d):" % len(HYPER_OPERATORS), OPERATOR_NAMES)
print("yaml operators: ['swap', '2opt_intra', 'relocate', 'swap_star', 'perturb']")

# tiny synthetic instance
rng = np.random.default_rng(0)
n = 9
coords = rng.uniform(0, 10, size=(n, 2))
dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
wastes = {i: float(rng.uniform(5, 40)) for i in range(1, n)}
capacity, R, C = 60.0, 1.0, 1.0
mandatory = [1, 2, 3]

from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
init = build_greedy_routes(dist_matrix=dist, wastes=wastes, capacity=capacity, R=R, C=C,
                           mandatory_nodes=mandatory, rng=__import__("random").Random(42))

# 1) params.operators restricted to 1 operator, yaml-style
p = HyperACOParams(operators=["swap"], n_ants=4, max_iterations=3, seed=42)
print("\nparams.operators passed in:", p.operators)
s = HyperHeuristicACO(dist, wastes, capacity, R, C, p, init, mandatory)
print("solver.operator_names (%d):" % len(s.operator_names), s.operator_names)
print("solver.sequence_length:", s.sequence_length, "(n_operators; yaml 'sequence_length: 5' has no effect)")
print("=> params.operators is IGNORED; all 11 operators always active:", set(p.operators) != set(s.operator_names))

# 2) determinism (seeds forwarded to both RNGs)
p2 = HyperACOParams(n_ants=6, max_iterations=8, time_limit=30, seed=42, profit_aware_operators=True)
r1 = HyperHeuristicACO(dist, wastes, capacity, R, C, p2, init, mandatory).solve()
r2 = HyperHeuristicACO(dist, wastes, capacity, R, C, p2, init, mandatory).solve()
print("\n=== [e] determinism with seed=42 ===")
print("run1 == run2:", r1 == r2, "| profit run1:", round(r1[1], 4), "profit run2:", round(r2[1], 4))

# 3) which operators actually execute at runtime (counts over a full solve)
counts = {k: 0 for k in HYPER_OPERATORS}
orig = dict(HYPER_OPERATORS)
for k, fn in orig.items():
    def make(k, fn):
        def wrapper(ctx):
            counts[k] += 1
            return fn(ctx)
        return wrapper
    HYPER_OPERATORS[k] = make(k, fn)
p3 = HyperACOParams(n_ants=8, max_iterations=15, time_limit=20, seed=42, profit_aware_operators=True)
best = HyperHeuristicACO(dist, wastes, capacity, R, C, p3, init, mandatory).solve()
for k in HYPER_OPERATORS:
    HYPER_OPERATORS[k] = orig[k]
print("\n=== [a] operator application counts over one solve (yaml lists only 5) ===")
for k, c in counts.items():
    flag = "" if k in ("swap","2opt_intra","relocate","swap_star","perturb") else "   <-- NOT in yaml list"
    print(f"  {k:15s} {c}{flag}")

# 4) virtual-start row: write-only? record rows read by _select_sequence
reads = []
class Spy(HyperHeuristicACO):
    def _select_sequence(self, start_op_idx):
        reads.append(start_op_idx)
        return super()._select_sequence(start_op_idx)
p4 = HyperACOParams(n_ants=6, max_iterations=6, time_limit=10, seed=42)
spy = Spy(dist, wastes, capacity, R, C, p4, init, mandatory)
vrow = spy.n_operators  # index of virtual row
spy.solve()
print("\n=== [b] pheromone deposit row vs selection rows ===")
print("selection start rows used (all must be < n_operators=%d):" % vrow, sorted(set(reads)))
print("row %d (virtual start) ever read by selection: %s" % (vrow, vrow in set(reads)))
print("tau[virtual row] after solve:", np.round(spy.tau[vrow], 3), "(> tau_0=1.0 means deposits landed on a never-read row)")
first_hop_unreinforced = not np.any(spy.tau[vrow] > spy.params.tau_0)
print("first-hop edges (start_op->first selected op) NEVER reinforced (deposit restarts at virtual row):", first_hop_unreinforced)
print("node-level SparsePheromoneTau after solve: stored edges =", len(spy.pheromone._pheromone),
      "default_value =", spy.pheromone.default_value, "(never deposited/evaporated on retained path)")

# 5) swallowed exception in apply_string_removal
print("\n=== [c] swallowed exception in removal wrappers ===")
d_small = np.zeros((3, 3))
ctx_bad = HyperOperatorContext(routes=[[9]], dist_matrix=d_small, waste={9: 1.0}, capacity=10.0,
                               R=1.0, C=1.0, rng=__import__("random").Random(0))
try:
    string_removal = __import__("logic.src.policies.helpers.operators", fromlist=["string_removal"]).string_removal
    string_removal(ctx_bad.routes, 1, ctx_bad.d, rng=ctx_bad.rng)
    print("raw string_removal: no error")
except Exception as e:
    print("raw string_removal raised:", type(e).__name__, e)
res = apply_string_removal(ctx_bad)
print("apply_string_removal same input -> returns:", res, "(exception silently swallowed, no traceback)")
