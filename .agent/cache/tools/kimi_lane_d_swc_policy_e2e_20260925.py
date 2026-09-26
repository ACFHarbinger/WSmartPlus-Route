import sys, numpy as np
sys.path.insert(0, ".")
from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.policy_swc_tcf import SWCTCFPolicy

# raw config shaped exactly like the simulator passes it (list-of-single-key-dicts yaml style)
raw_cfg = {
    "swc_tcf": {
        "gurobi": [
            {"Omega": 0.1}, {"delta": 0}, {"psi": 1},
            {"engine": "gurobi"}, {"framework": "gurobi"},
            {"mandatory_selection": {"other/ms_last_minute.yaml": ["last_minute_cf70"]}},
            {"vrpp": True}, {"time_limit": 60.0},
            {"route_improvement": {"other/ri_ftsp.yaml": ["default"]}},
        ]
    },
    "seed": 7,
}

class FakeBins:
    def __init__(self, c):
        self.c = np.asarray(c, dtype=float)

np.random.seed(0)
D = np.random.rand(8, 8) * 10 + 2.0
D = (D + D.T) / 2
np.fill_diagonal(D, 0.0)
D[0, :] = 4.0; D[:, 0] = 4.0; D[0, 0] = 0.0

policy = SWCTCFPolicy(config=raw_cfg)
tour, cost, profit, ctx, _ = policy.execute(
    mandatory=[1, 3, 5], bins=FakeBins([70, 20, 90, 10, 60, 30, 50]),
    distance_matrix=D, area="Rio Maior", waste_type="plastic",
    config=raw_cfg, seed=7,
)
print("E2E retained path: tour =", tour, " km =", round(cost, 2), " profit =", round(profit, 2))
assert tour[0] == 0 and tour[-1] == 0 and len(tour) > 2, "empty tour on retained path!"
assert {1, 3, 5}.issubset(set(tour)), "mandatory bins missing from tour!"

# now prove 'vrpp: false' is silently ignored on the typed-config path
raw_cfg2 = {
    "swc_tcf": {"gurobi": [{"Omega": 0.1}, {"delta": 0}, {"psi": 1},
                           {"engine": "gurobi"}, {"framework": "gurobi"},
                           {"vrpp": False}, {"time_limit": 60.0}]},
    "seed": 7,
}
p2 = SWCTCFPolicy(config=raw_cfg2)
cap, rev, cu, values = p2._load_area_params("Rio Maior", "plastic", raw_cfg2)
print("values keys contain 'vrpp'? ->", "vrpp" in values, "| values.get('vrpp') ->", values.get("vrpp"))
