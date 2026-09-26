"""Validate RCSPP DP (exact_mode) against brute force over all elementary routes."""
import sys, itertools
sys.path.insert(0, "/tmp/kimi-wsr")
import numpy as np
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.solver import RCSPPSolver

rng = np.random.default_rng(7)
fails = 0
for trial in range(30):
    n = rng.integers(4, 9)
    coords = rng.uniform(0, 50, size=(n + 1, 2))
    dm = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    w = {i: float(rng.uniform(1, 6)) for i in range(1, n + 1)}
    Q = float(rng.uniform(6, 12))
    R, C = 10.0, 1.0
    duals = {i: float(rng.uniform(0, 30)) for i in range(1, n + 1)}

    solver = RCSPPSolver(n, dm, w, Q, R, C, use_ng_routes=True, ng_neighborhood_size=n,
                         timeout=30.0, max_labels=10_000_000)
    routes = solver.solve(dual_values={"node_duals": duals}, max_routes=50, exact_mode=True)
    got_rc = routes[0].reduced_cost if routes else 0.0

    # brute force: all elementary routes, true RC = profit - sum duals
    best = 0.0
    nodes = list(range(1, n + 1))
    for r in range(1, len(nodes) + 1):
        for subset in itertools.combinations(nodes, r):
            if sum(w[i] for i in subset) > Q + 1e-9:
                continue
            for perm in itertools.permutations(subset):
                cost = dm[0, perm[0]] + sum(dm[a, b] for a, b in zip(perm, perm[1:])) + dm[perm[-1], 0]
                profit = sum(w[i] for i in perm) * R - C * cost
                rc = profit - sum(duals[i] for i in perm)
                best = max(best, rc)
    if abs(got_rc - best) > 1e-4:
        fails += 1
        print(f"trial {trial}: n={n} DP rc={got_rc:.6f} brute={best:.6f} DIFF={got_rc-best:+.6f}")
print(f"done: {fails}/30 mismatches (DP max-RC vs brute force over all elementary routes)")
