import itertools, sys, numpy as np
sys.path.insert(0, ".")
from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.dispatcher import run_swc_tcf_optimizer

def brute_force(fills, D, Q, R, C, Omega, mandatory):
    n = len(fills)
    best = None
    idx = list(range(1, n + 1))
    for r in range(1, n + 1):
        for subset in itertools.combinations(idx, r):
            if not all(m in subset for m in mandatory):
                continue
            if sum(fills[i - 1] for i in subset) > Q + 1e-9:
                continue
            for perm in itertools.permutations(subset):
                tour = (0,) + perm + (0,)
                km = sum(D[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))
                true_profit = R * sum(fills[i - 1] for i in subset) - C * km - Omega * 1
                if best is None or true_profit > best[0]:
                    best = (true_profit, subset, km)
    return best

print("=" * 70)
print("CHECK 1: 0.5*C factor in gurobi objective vs true (brute-force) optimum")
fills = [50, 99, 90]          # bin1 mandatory 50%, bin2 99%, bin3 90%
D = np.array([
    [0, 5, 15, 5],
    [5, 0, 15, 10],
    [15, 15, 0, 20],
    [5, 10, 20, 0],
], dtype=float)
values = {"Omega": 0.1, "delta": 0, "psi": 1, "Q": 150.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0}
# psi=1 => forced only if fill >= 100: none forced. delta=0 => all mandatory visited.
route, profit, cost = run_swc_tcf_optimizer(
    bins=np.array(fills, dtype=float), distance_matrix=D.tolist(), values=values,
    binsids=[1, 2, 3], mandatory_nodes=[1], number_vehicles=1,
    time_limit=60, framework="gurobi", optimizer="gurobi", seed=42)
print("gurobi route:", route, "model profit:", round(profit, 3), "km:", round(cost, 3))
collected = [x for x in route if x != 0]
true_profit = 1.0 * sum(fills[i - 1] for i in collected) - 1.0 * cost - 0.1
print("solution TRUE profit:", round(true_profit, 3), "collected:", collected)
bf = brute_force(fills, D, 150.0, 1.0, 1.0, 0.1, [1])
print("brute-force TRUE optimum: profit", round(bf[0], 3), "subset", bf[1], "km", round(bf[2], 3))
print(">> solver chose", collected, "but true optimum is", list(bf[1]),
      "-> travel cost halved in objective" if set(collected) != set(bf[1]) else "-> same choice")

print("=" * 70)
print("CHECK 2: unit divergence ortools(SCIP) vs native gurobi (realistic params)")
B, V = 19.0, 2.5
Q_pct = (3500.0 / (B * V)) * 100      # 7368.42 percent points
R_kg = 0.65 * 898 / 1000              # 0.5837 €/kg
R_scaled = R_kg * B * V / 100         # 0.27726 € per 1% fill
print(f"Q={Q_pct:.2f} percent, R_kg={R_kg:.4f} €/kg, R_scaled={R_scaled:.5f} €/%")
fills2 = [95.0] * 6
D2 = np.full((7, 7), 2.0); np.fill_diagonal(D2, 0); D2[0, :] = 5.0; D2[:, 0] = 5.0; D2[0, 0] = 0
values2 = {"Omega": 0.1, "delta": 0, "psi": 1, "Q": Q_pct, "R": R_scaled, "B": B, "C": 1.0, "V": V}
r_g, p_g, c_g = run_swc_tcf_optimizer(
    bins=np.array(fills2), distance_matrix=D2.tolist(), values=values2,
    binsids=list(range(1, 7)), mandatory_nodes=[1], number_vehicles=1,
    time_limit=120, framework="gurobi", optimizer="gurobi", seed=42)
col_g = [x for x in r_g if x != 0]
kg_g = sum(fills2[i-1]/100*B*V for i in col_g)
print(f"native gurobi: collected {len(col_g)} bins, model profit {p_g:.2f}, km {c_g:.1f}, kg {kg_g:.1f} (true cap 3500 kg = {Q_pct:.0f} %)")
r_o, p_o, c_o = run_swc_tcf_optimizer(
    bins=np.array(fills2), distance_matrix=D2.tolist(), values=values2,
    binsids=list(range(1, 7)), mandatory_nodes=[1], number_vehicles=1,
    time_limit=120, framework="ortools", optimizer="scip", seed=42)
col_o = [x for x in r_o if x != 0]
kg_o = sum(f/100*B*V for i in col_o for f in [fills2[i-1]])
print(f"ortools SCIP: collected {len(col_o)} bins, model profit {p_o:.2f}, km {c_o:.1f}, kg {kg_o:.1f} (its Q={Q_pct:.0f} read as KG)")
print("true revenue of one 95% bin:", round(R_kg*0.95*B*V, 3), "€; gurobi model rev/bin:", round(R_scaled*95, 3),
      "€; ortools model rev/bin:", round(R_scaled*0.95*B*V, 3), "€")

print("=" * 70)
print("CHECK 3: 6000 km arc filter divergence (km data, remote mandatory bins)")
D3 = np.array([[0, 7000, 7000], [7000, 0, 5], [7000, 5, 0]], dtype=float)
values3 = {"Omega": 0.1, "delta": 0, "psi": 1, "Q": 300.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0}
r3, p3, c3 = run_swc_tcf_optimizer(
    bins=np.array([50.0, 60.0]), distance_matrix=D3.tolist(), values=values3,
    binsids=[1, 2], mandatory_nodes=[1, 2], number_vehicles=1,
    time_limit=60, framework="gurobi", optimizer="gurobi", seed=42)
print("native gurobi on 7000-km instance:", r3, round(p3, 2), round(c3, 1))
r3o, p3o, c3o = run_swc_tcf_optimizer(
    bins=np.array([50.0, 60.0]), distance_matrix=D3.tolist(), values=values3,
    binsids=[1, 2], mandatory_nodes=[1, 2], number_vehicles=1,
    time_limit=60, framework="ortools", optimizer="scip", seed=42)
print("ortools SCIP on same instance:  ", r3o, round(p3o, 2), round(c3o, 1))

print("=" * 70)
print("CHECK 4: infeasible model -> silent empty tour")
values4 = {"Omega": 0.1, "delta": 0, "psi": 1, "Q": 150.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0}
# psi=1 forces both 100% bins; total fill 200 > Q=150 -> infeasible
r4, p4, c4 = run_swc_tcf_optimizer(
    bins=np.array([100.0, 100.0]), distance_matrix=[[0, 5, 5], [5, 0, 5], [5, 5, 0]], values=values4,
    binsids=[1, 2], mandatory_nodes=[], number_vehicles=1,
    time_limit=60, framework="gurobi", optimizer="gurobi", seed=42)
print("infeasible instance -> gurobi returned:", r4, p4, c4, "(silent empty tour, no exception)")

print("=" * 70)
print("CHECK 5: psi/delta actual semantics")
# psi=0.5 forces bins >=50% even when travel makes them a big loss
D5 = np.array([[0, 300], [300, 0]], dtype=float)
values5 = {"Omega": 0.1, "delta": 0, "psi": 0.5, "Q": 200.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0}
r5, p5, c5 = run_swc_tcf_optimizer(
    bins=np.array([60.0]), distance_matrix=D5, values=values5,
    binsids=[1], mandatory_nodes=[], number_vehicles=1,
    time_limit=30, framework="gurobi", optimizer="gurobi", seed=42)
print(f"psi=0.5, fill=60%, 300 km away -> route {r5} profit {p5:.1f} (bin FORCED at a loss of {60-600:.0f})")
# delta=1.0 makes mandatory coverage trivial -> mandatory bin skipped
values5b = {"Omega": 0.1, "delta": 1.0, "psi": 1, "Q": 200.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0}
r5b, p5b, c5b = run_swc_tcf_optimizer(
    bins=np.array([10.0]), distance_matrix=[[0, 400], [400, 0]], values=values5b,
    binsids=[1], mandatory_nodes=[1], number_vehicles=1,
    time_limit=30, framework="gurobi", optimizer="gurobi", seed=42)
print(f"delta=1.0, mandatory bin 400 km away -> route {r5b} profit {p5b:.1f} (mandatory SKIPPED)")
