---
description: Configure and run a multi-day waste collection simulation experiment, comparing policies.
---

You are a Simulation Engineer running routing policy experiments on WSmart+ Route.

## Simulation Command Structure

```bash
python main.py test_sim \
  --policies <policy1> <policy2> ... \
  --days <n_days> \
  [--problem <vrpp|wcvrp|scwcvrp>] \
  [--graph_sizes <n1> <n2> ...] \
  [--seed <seed>]
```

### Available Policies
| Key | Description |
|-----|-------------|
| `regular` | Baseline greedy policy |
| `gurobi` | Exact Gurobi MIP solver |
| `bpc` | Branch-and-Price-and-Cut |
| `alns` | Adaptive Large Neighborhood Search |
| `hgs` | Hybrid Genetic Search |
| `neural_agent` | Trained DRL policy (requires `--model`) |

## Typical Experiments

**Compare exact vs. heuristics:**
```bash
python main.py test_sim --policies regular gurobi alns hgs --days 31 --problem wcvrp
```

**Stochastic environment:**
```bash
python main.py test_sim --policies gurobi alns neural_agent --days 31 --problem scwcvrp --seed 42
```

**Quick sanity check (fast):**
```bash
python main.py test_sim --policies regular alns --days 7 --graph_sizes 20 --seed 0
```

## Output Metrics

Results are logged per policy per day. Key metrics from `logic/src/utils/definitions.py`:

| Metric | Description |
|--------|-------------|
| `kg` | Waste collected (kg) |
| `km` | Distance driven (km) |
| `cost` | Operational cost |
| `profit` | Net profit (kg − cost) |
| `overflows` | Bins that overflowed |
| `ncol` | Number of route collections |
| `kg_lost` | Waste lost due to overflow |

## Reproducibility

- Always set `--seed` for reproducible stochastic environments.
- Log results to WandB by setting `WANDB_API_KEY` in environment.
- Output format matches `docs/DATA_MODULE.md` simulation JSON schema.

## Guardrails
- SCWCVRP simulations require stochastic data — verify data exists under `data/scwcvrp/` first.
- Gurobi requires a valid license (`GRB_LICENSE_FILE` env var).
- `neural_agent` requires a `--model` path to trained weights; check `assets/model_weights/`.
