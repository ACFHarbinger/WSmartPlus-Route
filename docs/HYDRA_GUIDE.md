# Hydra Configuration Guide for WSmart-Route

> **Purpose**: Comprehensive guide to using Hydra configuration system in WSmart-Route
> **Last Updated**: 2026-02-08
> **Difficulty**: Beginner to Intermediate

---

## Table of Contents

1. [Introduction](#introduction)
2. [Configuration Composition](#configuration-composition)
3. [CLI Override Syntax](#cli-override-syntax)
4. [Multi-Run & Sweeps](#multi-run--sweeps)
5. [Available Config Groups](#available-config-groups)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)

---

## Introduction

WSmart-Route uses [Hydra](https://hydra.cc/) for configuration management, providing:

- **Composability**: Mix and match configurations from different groups
- **Reproducibility**: All experiment settings logged automatically
- **Sweeps**: Run hyperparameter searches with simple syntax
- **Type Safety**: Configs validated against Python dataclasses

### Why Two CLI Systems?

The codebase has a **dual CLI system** (see `main.py:212-256`):

| System | Commands | Status |
|--------|----------|--------|
| **Hydra** | `train`, `eval`, `test_sim`, `gen_data`, `mrl_train`, `hp_optim` | ✅ Primary |
| **Legacy** | `gui`, `test_suite`, `file_system` | ⚠️ Being migrated |

All new features use Hydra. Legacy commands will be migrated over time.

---

## Configuration Composition

### Composition Order

Hydra composes configs in this order (defined in `assets/configs/config.yaml`):

```yaml
defaults:
  - _self_           # 1. Base settings from config.yaml
  - tasks: train     # 2. Task-specific config (e.g., tasks/train.yaml)
  - envs: cwcvrp     # 3. Environment config (e.g., envs/cwcvrp.yaml)
  - model: am        # 4. Model architecture (e.g., model/am.yaml)
```

**Priority**: Later configs override earlier ones. So `model/am.yaml` can override settings from `tasks/train.yaml`.

### Example Composition

```bash
python main.py train  # Uses defaults: train + cwcvrp + am
```

This loads and merges:
1. `config.yaml` (base)
2. `tasks/train.yaml` (training parameters)
3. `envs/cwcvrp.yaml` (Capacitated Waste Collection VRP)
4. `model/am.yaml` (Attention Model architecture)

---

## CLI Override Syntax

### Basic Overrides

```bash
# Override single values
python main.py train seed=1234
python main.py train device=cpu

# Override nested values (dot notation)
python main.py train model.embed_dim=256
python main.py train rl.learning_rate=1e-4
python main.py train env.num_loc=100
```

### Config Group Selection

```bash
# Change model architecture
python main.py train model=tam  # Temporal Attention Model
python main.py train model=ptr  # Pointer Network

# Change environment
python main.py eval envs=vrpp  # Vehicle Routing with Profits
python main.py eval envs=sdwcvrp  # Stochastic Demand WCVRP

# Change task
python main.py task=eval  # Evaluation task
python main.py task=test_sim  # Simulation testing
```

### Multiple Overrides

```bash
# Combine overrides
python main.py train \
    model=tam \
    envs=sdwcvrp \
    seed=42 \
    model.n_encode_layers=6 \
    rl.batch_size=512
```

### List and Dict Overrides

```bash
# Override list elements
python main.py train env.graph_sizes='[20,50,100]'

# Override entire dicts
python main.py train 'rl.optimizer={name: adamw, lr: 1e-3}'
```

---

## Multi-Run & Sweeps

### Basic Multi-Run

Use `-m` or `--multirun` flag:

```bash
# Sweep over seeds
python main.py -m train seed=1,2,3,4,5

# Sweep over model dimensions
python main.py -m train model.embed_dim=64,128,256,512

# Combinatorial sweep (Cartesian product)
python main.py -m train \
    model.embed_dim=128,256 \
    rl.learning_rate=1e-3,1e-4 \
    seed=1,2,3
# Runs 2×2×3 = 12 experiments
```

### Range Sweeps

```bash
# Numeric ranges
python main.py -m train seed=range(1,11)  # Seeds 1-10
python main.py -m train rl.batch_size=range(128,513,128)  # 128,256,384,512

# Glob patterns (for sweeping over models/envs)
python main.py -m train model=glob(*)  # All models
python main.py -m train envs=glob(*)  # All environments
```

### Sweep Output Organization

Hydra creates separate directories for each run:

```
outputs/
├── 2026-02-08/
│   ├── 10-30-15/  # First run
│   │   ├── .hydra/
│   │   │   ├── config.yaml  # Full merged config
│   │   │   └── overrides.yaml  # CLI overrides
│   │   └── train.log
│   ├── 10-31-22/  # Second run
│   └── ...
└── multirun/
    └── 2026-02-08/
        ├── 10-35-00/
        │   ├── 0/  # seed=1
        │   ├── 1/  # seed=2
        │   └── 2/  # seed=3
        └── ...
```

---

## Available Config Groups

### Tasks (tasks/)

| Config | Command | Purpose |
|--------|---------|---------|
| `train.yaml` | `train` | RL training with PyTorch Lightning |
| `evaluation.yaml` | `eval` | Model evaluation on test sets |
| `test_sim.yaml` | `test_sim` | Multi-day simulator testing |
| `gen_data.yaml` | `gen_data` | Problem instance generation |
| `hpo.yaml` | `hp_optim` | Hyperparameter optimization (Optuna) |
| `meta_train.yaml` | `mrl_train` | Meta-RL cross-distribution training |

### Environments (envs/)

| Config | Problem | Description |
|--------|---------|-------------|
| `vrpp.yaml` | VRPP | Vehicle Routing with Profits |
| `cvrpp.yaml` | CVRPP | Capacitated VRPP |
| `wcvrp.yaml` | WCVRP | Waste Collection VRP |
| `cwcvrp.yaml` | CWCVRP | Capacitated WCVRP (default) |
| `sdwcvrp.yaml` | SDWCVRP | Stochastic Demand WCVRP |
| `scwcvrp.yaml` | SCWCVRP | Selective Capacitated WCVRP |

### Models (model/)

| Config | Architecture | Best For |
|--------|--------------|----------|
| `am.yaml` | Attention Model | General VRP tasks (default) |
| `tam.yaml` | Temporal AM | Multi-day problems (WCVRP) |
| `deep_decoder.yaml` | Deep Decoder AM | Large instances (100+ nodes) |
| `ptr.yaml` | Pointer Network | TSP, simpler routing |
| `symnco.yaml` | Symmetric NCO | Symmetric problems |
| `moe.yaml` | Mixture of Experts | Multi-distribution generalization |

### Policies (policies/)

Classical optimization policies (used in `test_sim`):

| Config | Solver | Type |
|--------|--------|------|
| `policy_alns.yaml` | ALNS | Metaheuristic |
| `policy_hgs.yaml` | HGS | Genetic |
| `policy_bcp.yaml` | Gurobi BCP | Exact (slow) |
| `policy_lkh.yaml` | Lin-Kernighan | TSP heuristic |
| `policy_neural.yaml` | Trained AM/TAM | Neural |

---

## Common Use Cases

### 1. Train a Model

```bash
# Default (AM on CWCVRP, 50 nodes)
python main.py train

# Custom configuration
python main.py train \
    model=tam \
    envs=sdwcvrp \
    env.num_loc=100 \
    rl.n_epochs=100 \
    rl.batch_size=256 \
    seed=42
```

### 2. Evaluate a Trained Model

```bash
# Evaluate on test set
python main.py eval \
    eval.checkpoint=assets/model_weights/best_model.ckpt \
    eval.data_path=data/vrpp/test.pkl \
    eval.decode_type=greedy

# Beam search decoding
python main.py eval \
    eval.checkpoint=... \
    eval.decode_type=beam_search \
    eval.beam_width=10
```

### 3. Run Simulator Test

```bash
# Compare policies over 31 days
python main.py test_sim \
    test.policies='[neural,gurobi,alns]' \
    test.days=31 \
    test.area=manhattan \
    test.seeds='[1,2,3,4,5]'
```

### 4. Generate Training Data

```bash
# Generate VRPP instances
python main.py gen_data \
    gen.problem=vrpp \
    gen.graph_sizes='[20,50,100]' \
    gen.samples_per_size=10000
```

### 5. Hyperparameter Optimization

```bash
# Optuna HPO with 50 trials
python main.py hp_optim \
    hpo.n_trials=50 \
    hpo.sampler=tpe \
    env.num_loc=50
```

### 6. Ablation Studies

```bash
# Compare encoder types
python main.py -m train \
    model.encoder_type=gat,gcn,tgc,mlp \
    seed=range(1,6)  # 4×5 = 20 runs

# Compare normalization strategies
python main.py -m train \
    model.normalization=batch,layer,instance,group \
    model.embed_dim=128,256
```

---

## Troubleshooting

### Common Errors

#### 1. "Could not find config group"

```bash
# Error
python main.py train model=wrong_name

# Fix: Check available models
ls assets/configs/model/*.yaml
python main.py train model=am  # Use correct name
```

#### 2. "Override  option already exists"

```bash
# Error: Duplicate override
python main.py train seed=42 seed=123

# Fix: Use last override or --config-path
python main.py train seed=123
```

#### 3. "Missing @dataclass field"

Hydra validates configs against `logic/src/configs.py`. If you add a new field to a config YAML, add it to the corresponding dataclass:

```python
# In logic/src/configs.py
@dataclass
class ModelConfig:
    embed_dim: int = 128
    new_field: int = 10  # Add this
```

### Debugging Tips

```bash
# Print full merged config without running
python main.py train --cfg job

# Print only overrides
python main.py train model=tam --cfg override

# Resolve all interpolations
python main.py train --cfg job --resolve
```

### Output Directory Conflicts

If Hydra complains about existing output directories:

```bash
# Option 1: Allow overwrites
python main.py train hydra.run.dir=./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}

# Option 2: Clear old outputs
rm -rf outputs/

# Option 3: Use unique run names
python main.py train +experiment_name=my_ablation_v2
```

---

## Next Steps

- **See [CLAUDE.md](../CLAUDE.md)** for coding standards and architecture
- **See [COMPATIBILITY.md](../COMPATIBILITY.md)** for model-environment compatibility matrix
- **See [README.md](../README.md)** for installation and quick start

---

## References

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Hydra Override Grammar](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [Hydra Multi-Run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
- [Structured Configs](https://hydra.cc/docs/tutorials/structured_config/intro/)
