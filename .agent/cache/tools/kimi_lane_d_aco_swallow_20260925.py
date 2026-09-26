import sys
sys.path.insert(0, "/tmp/kimi-wsr")
import numpy as np, random, traceback
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators import (
    HyperOperatorContext, apply_string_removal,
)
from logic.src.policies.helpers.operators import string_removal

d3 = np.zeros((3, 3))
try:
    string_removal([[8, 9]], 1, d3, rng=random.Random(0))
    print("raw: no error")
except Exception as e:
    print("raw string_removal([[8,9]],1,3x3):", type(e).__name__, "->", e)

ctx = HyperOperatorContext(routes=[[8, 9]], dist_matrix=d3, waste={8: 1.0, 9: 1.0},
                           capacity=10.0, R=1.0, C=1.0, rng=random.Random(0))
print("apply_string_removal same input ->", apply_string_removal(ctx), "(IndexError silently swallowed)")
