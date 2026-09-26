"""DS-34 check: HGS LinearSplit (limited and unlimited fleet) against brute force over all
contiguous route splits of a giant tour with optional skips.

usage (repo root): .venv/bin/python .agent/cache/tools/hgs_split_check.py
"""
import sys, os, itertools
sys.path.insert(0, os.getcwd())
import numpy as np
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.split import LinearSplit
rng = np.random.default_rng(0); bad = 0; tested = 0
for t in range(200):
    n = int(rng.integers(3, 7)); pts = rng.uniform(0, 10, size=(n + 1, 2)); dm = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    w = {i: float(rng.integers(1, 5)) for i in range(1, n + 1)}; Q = float(rng.integers(4, 9)); R = float(rng.uniform(1, 4))
    tour = list(rng.permutation(np.arange(1, n + 1)))
    un = LinearSplit(dm, w, Q, R, 1.0, max_vehicles=0).split(tour)
    for K in (1, 2, n):
        lim = LinearSplit(dm, w, Q, R, 1.0, max_vehicles=K).split(tour)
        tested += 1
        feas = all(sum(w[i] for i in r) <= Q + 1e-9 for r in lim[0]) and len(lim[0]) <= K
        if not feas or (K == n and abs(lim[1] - un[1]) > 1e-6):
            bad += 1
            if bad <= 5: print(f"t={t} K={K} n={n} unlimited={un[1]:.3f} {un[0]} limited={lim[1]:.3f} {lim[0]} feasible={feas}")
print(f"{tested - bad}/{tested} limited splits feasible and equal to unlimited when K=n")
# brute force: each node is skipped, starts a new route, or continues the current one
def brute(tour, dm, w, Q, R, K):
    best = 0.0
    n = len(tour)
    for labels in itertools.product((0, 1, 2), repeat=n):  # 0 skip, 1 new route, 2 continue
        routes, cur, ok = [], None, True
        for node, lab in zip(tour, labels):
            if lab == 0:
                cur = None; continue
            if lab == 1 or cur is None:
                if lab == 2: ok = False; break
                cur = [node]; routes.append(cur)
            else: cur.append(node)
        if not ok or len(routes) > K or any(sum(w[i] for i in r) > Q for r in routes): continue
        prof = sum(R * sum(w[i] for i in r) - sum(dm[a][b] for a, b in zip([0] + r, r + [0])) for r in routes)
        best = max(best, prof)
    return best
rng = np.random.default_rng(1); bad = 0
for t in range(150):
    n = int(rng.integers(3, 7)); pts = rng.uniform(0, 10, size=(n + 1, 2)); dm = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    w = {i: float(rng.integers(1, 5)) for i in range(1, n + 1)}; Q = float(rng.integers(4, 9)); R = float(rng.uniform(1, 4))
    tour = [int(x) for x in rng.permutation(np.arange(1, n + 1))]
    for K in (1, 2):
        got = LinearSplit(dm, w, Q, R, 1.0, max_vehicles=K).split(tour)[1]
        opt = brute(tour, dm, w, Q, R, K)
        if abs(got - opt) > 1e-6:
            bad += 1
            if bad <= 3: print("K", K, "got", got, "opt", opt)
print(f"brute-force K=1,2: {300 - bad}/300 optimal")
rng = np.random.default_rng(1); badu = 0
for t in range(150):
    n = int(rng.integers(3, 7)); pts = rng.uniform(0, 10, size=(n + 1, 2)); dm = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    w = {i: float(rng.integers(1, 5)) for i in range(1, n + 1)}; Q = float(rng.integers(4, 9)); R = float(rng.uniform(1, 4))
    tour = [int(x) for x in rng.permutation(np.arange(1, n + 1))]
    got = LinearSplit(dm, w, Q, R, 1.0, max_vehicles=0).split(tour)[1]
    if abs(got - brute(tour, dm, w, Q, R, n)) > 1e-6: badu += 1
print(f"brute-force unlimited: {150 - badu}/150 optimal")
