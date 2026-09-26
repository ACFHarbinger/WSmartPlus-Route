#!/usr/bin/env bash
# Acceptance smoke for the minimal export (run from the repo root or an extracted archive).
# gen_data -> train (tiny AM, gamma1) -> offline eval -> 10-day test_sim of all nine policies.
# Fails unless eval reports non-zero km/kg and every policy log routes (km > 0) on some day.
set -euo pipefail
PY="${PY:-.venv/bin/python}"
OUT="${1:-$(mktemp -d)}"
DAYS="${DAYS:-10}"
# SIM_CORES caps simulator worker processes (each loads torch/CUDA/Gurobi); the yaml default 0 means one per spare core.
mkdir -p "$OUT"
STAMP="$OUT/.stamp"; touch "$STAMP"
echo "[smoke] output: $OUT"

"$PY" main.py gen_data data.dataset_type=test_simulator data.problem=vrpp 'data.data_distributions=[gamma1]' \
  'data.graphs=[{num_loc: 20, n_days: 6, n_samples: 2}]' data.data_dir="$OUT/gen" data.name=smoke data.overwrite=true \
  > "$OUT/gen.log" 2>&1 || { tail -20 "$OUT/gen.log"; exit 1; }
DS="$(ls "$OUT"/gen/*.npz | head -1)"; echo "[smoke] gen_data ok: $DS"

"$PY" main.py train train.env.name=vrpp \
  'train.env.curriculum_graphs=[{num_loc: 20, n_samples: 64, area: riomaior, waste_type: plastic, distance_method: ogd, focus_size: 64, focus_graph: graphs_20V_1N_plastic.json, n_days: 1, load_dataset: null}]' \
  'train.env.eval_graphs=[]' train.batch_size=16 train.num_workers=0 train.policy.model.encoder.embed_dim=32 \
  train.policy.model.encoder.hidden_dim=64 train.policy.model.encoder.n_layers=1 train.policy.model.encoder.n_heads=2 \
  train.policy.model.decoder.n_heads=2 train.policy.mandatory_selection=null train.data_distribution=gamma1 \
  train.train_time=false train.model_weights_path="$OUT/ckpt" train.final_model_path="$OUT/am/epoch-1.pt" \
  > "$OUT/train.log" 2>&1 || { tail -20 "$OUT/train.log"; exit 1; }
test -f "$OUT/am/epoch-1.pt"; echo "[smoke] train ok"

"$PY" main.py eval eval.policy.model.load_path="$OUT/am" "eval.datasets=[$DS]" eval.val_size=12 eval.eval_batch_size=4 \
  eval.problem=vrpp eval.env.name=vrpp eval.results_dir="$OUT/eval" > "$OUT/eval.log" 2>&1 || { tail -20 "$OUT/eval.log"; exit 1; }
"$PY" - "$OUT/eval.log" <<'PYEOF'
import re, sys
txt = open(sys.argv[1]).read()
km = [float(x) for x in re.findall(r"(?i)\bkm\b[^0-9\-]*([0-9.]+)", txt)]
kg = [float(x) for x in re.findall(r"(?i)\bkg\b[^0-9\-]*([0-9.]+)", txt)]
assert km and max(km) > 0 and kg and max(kg) > 0, f"eval km/kg not positive: km={km[:3]} kg={kg[:3]}"
print(f"[smoke] eval ok: km={max(km):.3f} kg={max(kg):.3f}")
PYEOF

"$PY" main.py test_sim sim.graph.area=riomaior sim.graph.num_loc=20 sim.graph.n_days="$DAYS" \
  'sim.graph.dm_filepath="gmaps_distmat_plastic[riomaior].csv"' sim.graph.focus_graph=graphs_20V_1N_plastic.json \
  sim.graph.load_dataset=null sim.data_distribution=emp sim.cpu_cores="${SIM_CORES:-4}" p.na.na.amgat.0.model_path="$OUT/am" \
  > "$OUT/sim.log" 2>&1 || { tail -30 "$OUT/sim.log"; exit 1; }
"$PY" - "$STAMP" "$DAYS" <<'PYEOF'
import glob, json, os, sys
stamp, days = os.path.getmtime(sys.argv[1]), sys.argv[2]
logs = [f for f in glob.glob(f"assets/output/{days}days/**/log_*.json", recursive=True) if os.path.getmtime(f) > stamp]
pols = ["aco_hh", "alns", "bpc", "hgs", "_na_", "psoma", "pg_clns", "sans", "swc_tcf"]
bad = []
for f in sorted(logs):
    d = json.load(open(f))["daily"]["0"]
    km = max(d["km"])
    print(f"[smoke]   {os.path.basename(f):60s} max km/day {km:8.2f}")
    if km <= 0: bad.append(f)
missing = [p for p in pols if not any(p in os.path.basename(f) for f in logs)]
assert not missing, f"no log for: {missing}"
assert not bad, f"logs that never route: {bad}"
print(f"[smoke] test_sim ok: {len(logs)} logs, all route")
PYEOF
echo "[smoke] PASS"
