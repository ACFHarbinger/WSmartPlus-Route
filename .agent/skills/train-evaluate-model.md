---
description: Train a neural routing model, evaluate it, and compare against classical baselines.
---

You are an AI Research Scientist running training and evaluation experiments in WSmart+ Route.

## Training

```bash
# Standard REINFORCE training
python main.py train_lightning model=am env.name=vrpp env.num_loc=50

# With experiment config
python main.py train_lightning experiment=meta_rl model=am env.name=vrpp train.n_epochs=50

# Hyperparameter optimization sweep
python main.py train_lightning experiment=hpo env.name=wcvrp
```

### Key Hydra Config Overrides

| Override | Values | Effect |
|----------|--------|--------|
| `model` | `am`, `tam`, `ddam` | Architecture |
| `env.name` | `vrpp`, `wcvrp`, `scwcvrp` | Problem type |
| `env.num_loc` | `20`, `50`, `100`, `150` | Graph size |
| `train.n_epochs` | integer | Training duration |
| `train.batch_size` | integer | Batch size (tune for VRAM) |
| `train.lr` | float | Learning rate |

### Config Sanitization (CRITICAL — AGENTS.md §6.1)
When passing Hydra configs to Lightning modules:
```python
# CORRECT
from logic.src.utils.configs.setup_utils import deep_sanitize
common_kwargs = deep_sanitize(cfg.rl)
common_kwargs["env"] = env   # inject after sanitize
model = MyModule(**common_kwargs)

# WRONG — DictConfig causes YAML errors in Lightning hparams
model = MyModule(**cfg.rl)
```

## Evaluation

```bash
# Evaluate a trained model on test data
python main.py eval data/vrpp/test.pkl --model ./assets/model_weights/best.pt

# Generate test data first if needed
python main.py gen_data test --problem vrpp --graph_sizes 20 50 --seed 1234
```

## Comparing Against Baselines

Run simulation with neural agent alongside classical solvers:
```bash
python main.py test_sim \
  --policies neural_agent gurobi alns hgs \
  --model ./assets/model_weights/best.pt \
  --days 31 --problem vrpp --seed 42
```

## Batch Size Guidelines (VRAM)

| VRAM | `env.num_loc=50` | `env.num_loc=100` |
|------|-----------------|------------------|
| 8 GB | 64–128 | 32–64 |
| 12 GB | 256 | 128 |
| 24 GB | 512–1024 | 256–512 |

## Experiment Tracking

Set `WANDB_API_KEY` to log runs to Weights & Biases automatically.

## Guardrails
- Use `uv run` prefix or activate `.venv` before any `python main.py` command.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for large graph training.
- Model weights must go under `assets/model_weights/` — do NOT commit large files; use Git LFS.
- Do NOT push to `main` branch directly; all changes require PR review (AGENTS.md §5.3).
