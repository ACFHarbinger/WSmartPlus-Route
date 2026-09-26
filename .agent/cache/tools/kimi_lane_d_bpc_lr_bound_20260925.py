import sys, time, logging
sys.path.insert(0, "/tmp/kimi-wsr")
import numpy as np
logging.basicConfig(level=logging.CRITICAL)

print("=" * 70)
print("TEST 2 (fixed seed): LR bound vs z* (multi-route)")
print("=" * 70)
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.params import BPCParams
from logic.src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning import compute_lr_bound_at_node

# 2-node instance: z* = 116 needs two routes
dm2 = np.array([[0, 1, 1], [1, 0, 100], [1, 100, 0]], dtype=float)
w2 = {1: 6.0, 2: 6.0}
lr_params = BPCParams(lr_pre_pruning=True, lr_lambda_init=0.0, lr_max_subgradient_iters=10,
                      lr_subgradient_theta=1.0, lr_op_time_limit=3.0, optimality_gap=0.005, seed=42)
ub, lam_star, visited = compute_lr_bound_at_node(dist_matrix=dm2, wastes=w2, capacity=10.0, R=10.0, C=1.0,
                                                 mandatory=set(), forced_out=set(), params=lr_params,
                                                 time_budget=10.0, env=None, recorder=None)
print(f"LR ub = {ub:.4f}, z* = 116.0  -> bound valid (ub>=z*)? {ub >= 116 - 1e-6}; pruned at incumbent>=58? {ub <= 58*1.005}")

print()
print("=" * 70)
print("TEST LR end-to-end, cutting_planes='sri' (no LCI engines)")
print("=" * 70)
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import run_bpc
dm = np.array([[0, 1, 1, 1],
               [1, 0, 5, 1],
               [1, 5, 0, 1],
               [1, 1, 1, 0]], dtype=float)
wastes = {1: 1.0, 2: 1.0, 3: 1.0}
base = dict(time_limit=30.0, max_bb_nodes=500, max_cg_iterations=100, optimality_gap=1e-6,
            early_termination_gap=1e-9, rc_tolerance=1e-9, cutting_planes="sri",
            branching_strategy="divergence", search_strategy="depth_first",
            enable_node_visitation_branching=True, profit_aware_operators=True,
            use_ng_routes=True, ng_neighborhood_size=8, exact_mode=False,
            enable_strong_branching_heuristic=True, seed=42)
r1, o1 = run_bpc(dm, wastes, 2.0, 10.0, 1.0, BPCParams(**{**base, "lr_pre_pruning": False}))
print(f"lr=False: {r1} obj={o1:.4f} (z*=25)")
r2, o2 = run_bpc(dm, wastes, 2.0, 10.0, 1.0, BPCParams(**{**base, "lr_pre_pruning": True,
                                                            "lr_lambda_init": 0.0, "lr_max_subgradient_iters": 40,
                                                            "lr_op_time_limit": 5.0, "lr_warm_start_cg": True}))
print(f"lr=True : {r2} obj={o2:.4f} (z*=25)  -> {'SUBOPTIMAL (invalid LR bound pruned optimum)' if o2 < 25 - 1e-6 else 'optimal'}")

print()
print("=" * 70)
print("TEST invalid PhysicalCapacityLCI cut demonstrated directly")
print("=" * 70)
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.common import Route
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel
from logic.src.policies.helpers.solvers_and_matheuristics.search.cutting_planes import PhysicalCapacityLCIEngine
vm = VRPPModel(4, dm, wastes, 2.0, revenue_per_kg=10.0, cost_per_km=1.0)
m = VRPPMasterProblem(3, set(), dm, wastes, 2.0, 10.0, 1.0)
cols = [Route([3, 1], 3.0, 20.0, 2.0, {3, 1}), Route([2], 2.0, 10.0, 1.0, {2})]
m.build_model(cols)
m.solve_lp_relaxation()
# simulate fractional y: x31=1, x2=1 gives y=(1,1,1); force by solving first
print("y_vals:", m.get_node_visitation())
# manually craft fractional point: x31=.5 x2=1 x3(unused)... use engine with a fake LP point is complex;
# instead show the cut the engine adds for cover {1,2,3}:
eng = PhysicalCapacityLCIEngine(vm)
added = eng.separate_and_add_cuts(m, 5)
print("cuts added:", added)
for c in m.model.getConstrs():
    print("  ", c.ConstrName, c.Sense, c.RHS)
print("Cut semantics: sum over routes of (#cover nodes visited) * lambda <= 2")
print("  -> forbids x31=1 + x2=1 (visits nodes 1,2,3 with TWO routes, each feasible): INVALID for multi-route SP")
