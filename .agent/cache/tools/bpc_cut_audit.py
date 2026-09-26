"""Audit which BPC cut (if any) removes the brute-force optimum of one generated instance.

usage (repo root): .venv/bin/python .agent/cache/tools/bpc_cut_audit.py <mode> <instance_index> [n_nodes] [seed]
(mode/instance as printed by bpc_bruteforce_check.py)
"""
import itertools
import logging
import os
import sys

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(__file__))
logging.disable(logging.WARNING)
import numpy as np
from omegaconf import OmegaConf

from bpc_bruteforce_check import partitions, route_cost
import logic.src.policies.helpers.solvers_and_matheuristics.search.cutting_planes as CP
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import run_bpc
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.params import BPCParams

mode, target = sys.argv[1], int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
for k in range(target + 1):
    pts = rng.uniform(0, 10, size=(n + 1, 2)); dm = np.round(np.linalg.norm(pts[:, None] - pts[None], axis=-1), 2)
    w = {i: float(rng.integers(1, 5)) for i in range(1, n + 1)}; Q, R, C = float(rng.integers(4, 9)), float(rng.uniform(1, 4)), 1.0
    mand = frozenset(int(i) for i in rng.choice(np.arange(1, n + 1), size=2, replace=False)) if mode == "mandatory" else frozenset()
    vl = int(rng.integers(1, 3)) if mode == "fleet" else None
best = (-float("inf"), None)
for r in range(len(mand), n + 1):
    for sub in itertools.combinations(range(1, n + 1), r):
        if not mand <= set(sub): continue
        for part in partitions(list(sub)):
            if any(sum(w[i] for i in rt) > Q for rt in part) or (vl is not None and len(part) > vl): continue
            pr = sum(R * w[i] for rt in part for i in rt) - C * sum(route_cost(dm, tuple(rt)) for rt in part)
            if pr > best[0]: best = (pr, part)
OPT = [frozenset(rt) for rt in best[1]]
print(f"w={w} Q={Q} R={R:.3f} mand={sorted(mand)} K={vl} optimum={best[0]:.4f} routes={best[1]}")

def wrap(cls):
    orig = cls.separate_and_add_cuts
    def sep(self, master, max_cuts, **kw):
        master.model.update(); before = {c.ConstrName for c in master.model.getConstrs()}
        out = orig(self, master, max_cuts, **kw); master.model.update()
        name_to_idx = {lv.VarName: j for j, lv in enumerate(master.lambda_vars)}
        for c in master.model.getConstrs():
            if c.ConstrName in before: continue
            row = master.model.getRow(c); terms = []; opt_coef = {}
            for i in range(row.size()):
                j = name_to_idx.get(row.getVar(i).VarName)
                if j is None: continue
                rt = frozenset(master.routes[j].nodes); coef = row.getCoeff(i)
                terms.append(f"{coef:g}*{tuple(master.routes[j].nodes)}")
                if rt in OPT: opt_coef[rt] = coef  # one column per optimal route (orders share a set)
            lhs = sum(opt_coef.values())
            viol = (c.Sense == "<" and lhs > c.RHS + 1e-6) or (c.Sense == ">" and lhs < c.RHS - 1e-6)
            if viol:
                print(f"  VIOLATES OPTIMUM [{cls.__name__}] {c.ConstrName}: {' + '.join(terms[:8])}{' ...' if len(terms) > 8 else ''} {c.Sense}= {c.RHS:g} (optimum lhs {lhs:g})")
        return out
    cls.separate_and_add_cuts = sep
for name in dir(CP):
    obj = getattr(CP, name)
    if isinstance(obj, type) and name.endswith("Engine") and hasattr(obj, "separate_and_add_cuts") and name not in ("CuttingPlaneEngine", "CompositeCuttingPlaneEngine"):
        wrap(obj)
import logic.src.policies.helpers.solvers_and_matheuristics.pricing.solver as PS
import logic.src.policies.helpers.solvers_and_matheuristics.master_problem.model as MM
def path_cost(nodes):
    pth = (0,) + tuple(nodes) + (0,)
    return sum(dm[a][b] for a, b in zip(pth, pth[1:]))
def best_order_cost(nodes):
    return min(path_cost(p) for p in itertools.permutations(nodes))
_o_solve = PS.RCSPPSolver.solve
def _solve(self, *a, **kw):
    dv = kw.get("dual_values", a[0] if a else {})
    out = _o_solve(self, *a, **kw)
    nd = dv.get("node_duals", dv) if isinstance(dv, dict) else {}
    vdual = dv.get("vehicle_limit", 0.0) if isinstance(dv, dict) else 0.0
    cuts = [k for k, v in dv.items() if k not in ("node_duals", "vehicle_limit") and v] if isinstance(dv, dict) else []
    rcs = {tuple(sorted(rt)): round(R * sum(w[i] for i in rt) - C * best_order_cost(tuple(rt)) - sum(nd.get(i, 0.0) for i in rt) - vdual, 3) for rt in OPT}
    print(f"  pricing farkas={kw.get('is_farkas')} rc(optimal routes, node duals only)={rcs} cut-duals={cuts} returned={[(r.nodes, round(r.reduced_cost or 0, 3)) for r in out[:3]]}")
    return out
PS.RCSPPSolver.solve = _solve
_o_lp = MM.VRPPMasterProblem.solve_lp_relaxation
def _lp(self, *a, **kw):
    r = _o_lp(self, *a, **kw)
    for v, rt in zip(self.lambda_vars, self.routes):
        true = R * sum(w[i] for i in rt.nodes) - C * path_cost(rt.nodes)
        if getattr(self, "phase", 2) == 2 and abs(v.Obj - true) > 1e-6:
            print(f"  COLUMN OBJ MISMATCH {tuple(rt.nodes)} obj={v.Obj:.4f} true={true:.4f}")
    print(f"  LP={r[0] if isinstance(r, tuple) else r} phase={getattr(self, 'phase', '?')} n_cols={len(self.routes)}")
    return r
MM.VRPPMasterProblem.solve_lp_relaxation = _lp
cfg = OmegaConf.load("logic/configs/policies/policy_bpc.yaml"); flat = {}
for item in cfg.bpc.custom: flat.update(OmegaConf.to_container(item))
routes, obj = run_bpc(dm, w, Q, R, C, BPCParams.from_config({**flat, "time_limit": 30.0}), mandatory_indices=set(mand) or None, vehicle_limit=vl)
print(f"bpc={obj:.4f} routes={routes}")
