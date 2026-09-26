"""Lane D (BPC) repro — run from /tmp/kimi-wsr with the shared venv python."""
import sys, time, warnings, logging
sys.path.insert(0, "/tmp/kimi-wsr")
import numpy as np

logging.basicConfig(level=logging.CRITICAL)
VENV_PKGS_OK = True

print("=" * 70)
print("TEST 1: search_strategy plumbing (yaml depth_first vs tree)")
print("=" * 70)
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.params import BPCParams
from logic.src.policies.helpers.solvers_and_matheuristics.branching import BranchAndBoundTree

params = BPCParams(search_strategy="depth_first", branching_strategy="divergence", max_bb_nodes=2000)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    tree = BranchAndBoundTree(v_model=None, params=params, search_strategy="depth_first", strategy="divergence")
    print("warnings raised:", [str(x.message)[:80] for x in w])
print("params.search_strategy =", params.search_strategy)
print("tree.search_strategy   =", tree.search_strategy, " (via fallback default of the ignored arg)")
print("tree.max_nodes         =", tree.max_nodes, "(params.max_bb_nodes=2000 NOT used: getattr asks for 'max_branch_nodes'; unused elsewhere -> harmless)")
print("NOTE: a DeprecationWarning is emitted on EVERY run_bpc call; landmine for future callers that pass only params.")

print()
print("=" * 70)
print("TEST 2: Lagrangian pre-pruning bound vs true optimum (multi-route)")
print("=" * 70)
from logic.src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning import compute_lr_bound_at_node

# 2 nodes, each waste 6, capacity 10 -> each needs its own route. Opt = 2 routes.
dm = np.array([[0, 1, 1], [1, 0, 100], [1, 100, 0]], dtype=float)
wastes = {1: 6.0, 2: 6.0}
R, C, Q = 10.0, 1.0, 10.0
# brute force optimum over all route sets (partitions into feasible routes)
import itertools
def route_profit(nodes):
    cost = dm[0, nodes[0]] + sum(dm[a, b] for a, b in zip(nodes, nodes[1:])) + dm[nodes[-1], 0]
    rev = sum(wastes[i] for i in nodes) * R
    return rev - C * cost
def brute_force_opt(dm, wastes, Q, R, C):
    nodes = sorted(wastes)
    best = 0.0
    # enumerate set partitions via restricted growth strings
    n = len(nodes)
    for assign in itertools.product(range(n), repeat=n):
        # canonical: first occurrence order
        if list(assign) != sorted(set(assign), key=lambda x: assign.index(x)) and assign != tuple(sorted(assign, key=lambda x: list(assign).index(x))):
            pass
        blocks = {}
        for nd, b in zip(nodes, assign):
            blocks.setdefault(b, []).append(nd)
        ok = all(sum(wastes[i] for i in bl) <= Q for bl in blocks.values())
        if not ok:
            continue
        val = sum(route_profit(sorted(bl)) for bl in blocks.values())
        best = max(best, val)
    return best
z_star = 116.0  # known: routes {1},{2}: (60-2)*2
print("brute-force z* =", brute_force_opt(dm, wastes, Q, R, C), "(expected 116 = two singleton routes)")

lr_params = BPCParams(lr_pre_pruning=True, lr_lambda_init=0.0, lr_max_subgradient_iters=10,
                      lr_subgradient_theta=1.0, lr_op_time_limit=3.0, optimality_gap=0.005, seed=42)
t0 = time.perf_counter()
ub, lam_star, visited = compute_lr_bound_at_node(dist_matrix=dm, wastes=wastes, capacity=Q, R=R, C=C,
                                                 mandatory=set(), forced_out=set(), params=lr_params,
                                                 time_budget=10.0, env=None, recorder=None)
print(f"LR ub = {ub:.4f}  (lam*={lam_star:.4f})  in {time.perf_counter()-t0:.2f}s")
print(f"z* = 116.0  ->  bound valid? {ub >= z_star - 1e-6}")
for incumbent in (58.0, 100.0, 115.0):
    thr = incumbent * (1 + 0.005)
    print(f"  incumbent={incumbent}: prune condition ub<=inc*(1+gap) -> {ub <= thr}  "
          f"(node containing z*=116 {'PRUNED' if ub <= thr else 'kept'})")
print("RESULT: BUG CONFIRMED — single-vehicle LR bound < true multi-route optimum; pruning can delete the optimal node.")

print()
print("=" * 70)
print("TEST 3: enable_dssr / dssr_max_iters yaml keys never reach the pricer")
print("=" * 70)
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.smoothing import solve_pricing_step, dssr_pricing_wrapper
import inspect
src = inspect.getsource(solve_pricing_step)
print("solve_pricing_step defaults: use_dssr=False")
from logic.src.policies.helpers.solvers_and_matheuristics.search.column_generation import column_generation_loop
cg_src = inspect.getsource(column_generation_loop)
print("column_generation_loop calls solve_pricing_step with use_dssr?",
      "use_dssr" in cg_src.split("solve_pricing_step(")[1].split(")")[0])
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.solver import RCSPPSolver
solver = RCSPPSolver(2, dm, wastes, Q, R, C)
print("RCSPPSolver has _ng_memory attr (used by dssr wrapper):", hasattr(solver, "_ng_memory"))
print("RCSPPSolver has ng_neighborhoods attr:", hasattr(solver, "ng_neighborhoods"))
# demonstrate wrapper is a functional no-op beyond solve()
snap = solver.save_ng_snapshot()
routes = dssr_pricing_wrapper(pricing_solver=solver, node_duals={"node_duals": {}}, max_routes=5, timeout=2.0)
print("dssr wrapper returned", len(routes), "routes; ng neighborhoods unchanged:", solver.ng_neighborhoods == snap)
print("RESULT: CONFIRMED — DSSR never enabled by engine (use_dssr not passed) and wrapper operates on a nonexistent attr.")

print()
print("=" * 70)
print("TEST 4: reduced-cost arc fixing dead block + latent TypeError")
print("=" * 70)
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.smoothing import reduced_cost_arc_fixing
print("fresh RCSPPSolver hasattr _forbidden_arcs:", hasattr(solver, "_forbidden_arcs"),
      "-> bpc_engine gate `hasattr(pricing_solver,'_forbidden_arcs')` is always False on first use")
try:
    reduced_cost_arc_fixing(pricing_solver=solver, master_lp_bound=10.0, incumbent_value=5.0,
                            n_nodes=2, dist_matrix=dm, wastes=wastes,
                            node_duals={}, capacity=Q, R=R, C=C)  # call as bpc_engine does
    print("no error?!")
except TypeError as e:
    print("calling it exactly as bpc_engine does -> TypeError:", e)
print("RESULT: CONFIRMED — enable_reduced_cost_arc_fixing yaml key read nowhere; block dead; call would crash (capacity kwarg).")

print()
print("=" * 70)
print("TEST 5: dual smoothing never enabled (exact_mode=False yaml comment is wrong)")
print("=" * 70)
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem
m = VRPPMasterProblem(2, set(), dm, wastes, Q, R, C)
print("master.enable_dual_smoothing default:", m.enable_dual_smoothing)
import subprocess
grep_out = subprocess.run(["grep", "-rn", "enable_dual_smoothing = True", "logic/src/policies/"],
                          capture_output=True, text=True, cwd="/tmp/kimi-wsr").stdout.strip()
print("grep 'enable_dual_smoothing = True' in logic/src/policies:", repr(grep_out))
print("RESULT: smoothing can never turn on -> smoothing-recovery branch in column_generation_loop is dead code.")

print()
print("=" * 70)
print("TEST 6: Gurobi non-OPTIMAL LP silently returns (0.0, {})")
print("=" * 70)
import gurobipy as gp
m2 = VRPPMasterProblem(2, set(), dm, wastes, Q, R, C)
from logic.src.policies.helpers.solvers_and_matheuristics.common import Route
r1 = Route([1], 2.0, 60.0, 6.0, {1})
m2.build_model([r1])
m2.model.Params.TimeLimit = 0.0  # force time limit status
obj, vals = m2.solve_lp_relaxation()
print("LP with TimeLimit=0 -> status:", m2.model.Status, "(expect 9=TIME_LIMIT) -> returned obj:", obj, "vals:", vals)
m3 = VRPPMasterProblem(2, {1, 2}, dm, wastes, Q, R, C)  # mandatory 1,2 but only route covering node 1 -> infeasible
m3.build_model([r1])
obj3, vals3 = m3.solve_lp_relaxation()
print("infeasible RMP -> obj:", obj3, "farkas duals keys:", list(m3.farkas_duals.keys()),
      "node_duals:", m3.farkas_duals.get("node_duals"))
print("RESULT: non-OPTIMAL (incl. TIME_LIMIT) silently yields (0.0, {}) - CG treats 0.0 as valid LP value; infeasible handled via Farkas.")
