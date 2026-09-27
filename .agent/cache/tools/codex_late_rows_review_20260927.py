"""Read-only witnesses for the paper late-row review; no solver/simulator runs."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
result = {}
for suffix in ('', '_90d'):
    path = ROOT / ('docs/private/global/simulation/simulation_summary' + suffix + '.csv')
    rows = list(csv.DictReader(path.open()))
    result[path.name] = {'rows': len(rows), 'constructors': dict(Counter(r['constructor'] for r in rows))}
# Q14 proposed pair has an infeasible interval unless masses have a declared lattice.
E, epsilon, upper = 100.0, 0.01, 110.0
cases = []
for level in (E-epsilon, E-epsilon/2, E, E+epsilon):
    allowed = [o for o in (0,1) if level-E+epsilon <= upper*o+1e-12 and level >= E*o]
    cases.append({'content': level, 'allowed_binary_values': allowed})
assert cases[1]['allowed_binary_values'] == []
assert cases[2]['allowed_binary_values'] == [1]
result['epsilon_pair'] = {'capacity':E, 'epsilon':epsilon, 'cases':cases}
# The old single inequality also permits o=1 at below-capacity levels.
result['old_overflow_single_inequality'] = {'content':99, 'o1_is_feasible':99-100 <= 110}
# Feasible slide-28 trajectory after Q10 drops the zero-overflow cap.
# All start-of-period fills < psi*E except bin 3 on day 3; serve it then.
stock = [150,60,200,90]
arrival = [60,20,70,25]
records=[]; collected=0
for day in range(1,4):
    chosen = [2] if day == 3 else []
    assert all(i in chosen for i,v in enumerate(stock) if v >= 300)
    mass=sum(stock[i] for i in chosen);assert mass<=600
    records.append({'day':day,'start_mass':stock[:],'served_bins':[i+1 for i in chosen],'kg':mass})
    collected+=mass
    stock=[(0 if i in chosen else v)+arrival[i] for i,v in enumerate(stock)]
objective=(collected+sum(stock))*.0952 - 2*9.1 - .1
assert abs(objective-79.28)<1e-8
result['slide_example_Q10_rho_equals_R_witness']={'trajectory':records,'terminal_mass':stock,'objective':objective,'old_slide_rescored_trajectory':71.48,'meaning':'Feasible improving witness, not a claim of global optimality.'}
for rel in ('data/simulator/distance_matrix/submatrix/gmaps_distmat170_plastic[riomaior].csv',
            'data/simulator/distance_matrix/submatrix/gmaps_distmat100_plastic[riomaior].csv',
            'assets/output/30days/figueiradafoz350_plastic/gamma3/lm_ftsp/osm_distmat.csv'):
    matrix=np.loadtxt(ROOT/rel,delimiter=',')
    if matrix.shape[0]==matrix.shape[1]+1:matrix=matrix[1:]
    assert matrix.shape[0]==matrix.shape[1]
    sym=(matrix+matrix.T)/2
    n=matrix.shape[0]-1;off=~np.eye(n,dtype=bool)
    result[rel]={'directed_depot_outbound_median':float(np.median(matrix[0,1:])),
                 'directed_interbin_median':float(np.median(matrix[1:,1:][off])),
                 'symmetric_depot_median':float(np.median(sym[0,1:])),
                 'symmetric_interbin_median':float(np.median(sym[1:,1:][off]))}
path=ROOT/'.agent/cache/codex_late_rows_review_20260927.json'
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
