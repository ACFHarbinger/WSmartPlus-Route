"""D1 check: BPC with the shipped policy_bpc.yaml must match a brute-force VRPP optimum
(unlimited fleet, capacity Q, profit = R * collected - C * distance) on small random instances.

usage (repo root): .venv/bin/python .agent/cache/tools/bpc_bruteforce_check.py [n_instances] [n_nodes] [plain|mandatory|fleet] [seed]
"""
import itertools
import logging
import os
import sys

sys.path.insert(0, os.getcwd())
logging.disable(logging.WARNING)
import numpy as np
from omegaconf import OmegaConf

from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import run_bpc
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.params import BPCParams


def route_cost(dm, nodes):
    best = float("inf")
    for perm in itertools.permutations(nodes):
        path = (0,) + perm + (0,)
        best = min(best, sum(dm[a][b] for a, b in zip(path, path[1:])))
    return best


def partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in partitions(rest):
        yield [[first]] + p
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]


def brute_force(dm, w, Q, R, C, mandatory=frozenset(), vehicle_limit=None):
    n = len(w)
    best = -float("inf") if mandatory else 0.0
    for r in range(len(mandatory), n + 1):
        for subset in itertools.combinations(range(1, n + 1), r):
            if not mandatory <= set(subset):
                continue
            for part in partitions(list(subset)):
                if any(sum(w[i] for i in route) > Q + 1e-9 for route in part):
                    continue
                if vehicle_limit is not None and len(part) > vehicle_limit:
                    continue
                profit = sum(R * w[i] - 0 for route in part for i in route) - C * sum(route_cost(dm, tuple(rt)) for rt in part)
                best = max(best, profit)
    return best


def main():
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    mode = sys.argv[3] if len(sys.argv) > 3 else "plain"  # plain | mandatory | fleet
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    cfg = OmegaConf.load("logic/configs/policies/policy_bpc.yaml")
    flat = {}
    for item in cfg.bpc.custom:
        flat.update(OmegaConf.to_container(item))
    params = BPCParams.from_config({**flat, "time_limit": 30.0})
    rng = np.random.default_rng(seed)
    bad = 0
    for k in range(n_inst):
        pts = rng.uniform(0, 10, size=(n + 1, 2))
        dm = np.round(np.linalg.norm(pts[:, None] - pts[None], axis=-1), 2)
        w = {i: float(rng.integers(1, 5)) for i in range(1, n + 1)}
        Q, R, C = float(rng.integers(4, 9)), float(rng.uniform(1, 4)), 1.0
        mand = frozenset(int(i) for i in rng.choice(np.arange(1, n + 1), size=2, replace=False)) if mode == "mandatory" else frozenset()
        vl = int(rng.integers(1, 3)) if mode == "fleet" else None
        opt = brute_force(dm, w, Q, R, C, mand, vl)
        if opt == -float("inf"):
            continue
        _, obj = run_bpc(dm, w, Q, R, C, params, mandatory_indices=set(mand) or None, vehicle_limit=vl)
        ok = abs(obj - opt) < 1e-4 * max(1.0, abs(opt))
        bad += not ok
        print(f"[{mode}] instance {k:2d} mand={sorted(mand)} K={vl}: brute force {opt:9.4f}  bpc {obj:9.4f}  {'ok' if ok else 'MISMATCH'}", flush=True)
    print(f"{n_inst - bad}/{n_inst} match")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
