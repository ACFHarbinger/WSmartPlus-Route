# Migration Plan: Old RL Pipeline → New Lightning Pipeline (Full Deprecation)

> **Version**: 1.0
> **Created**: January 21, 2026
> **Goal**: Complete migration to Lightning pipeline and removal of old `reinforcement_learning/` directory
> **Estimated Effort**: 2-3 weeks

---

## Executive Summary

This document provides a comprehensive plan to fully migrate from the old reinforcement learning pipeline (`logic/src/pipeline/reinforcement_learning/`) to the new PyTorch Lightning-based pipeline (`logic/src/pipeline/rl/`), then safely remove the old implementation.

### Migration Scope

| Category | Old Pipeline | New Pipeline | Migration Status |
|----------|--------------|--------------|------------------|
| **RL Algorithms** | 5 | 10 | ✅ Superset |
| **Baselines** | 6 | 6 | ✅ Parity |
| **Meta-Learning** | 5 strategies | 6 strategies | ✅ Superset |
| **HPO** | 6 methods | 5 methods | ✅ Adequate (DEAP deprecated) |
| **CLI Arguments** | 150+ | Hydra Config | ✅ Hydra Config Adopted |
| **Vectorized Policies** | 3 files | 3 files (relocated) | ✅ Done |
| **Tests** | 12 files | Need migration | ✅ Tests Compatible |

---

## Table of Contents

1. [Current State: Import Dependencies](#1-current-state-import-dependencies)
2. [Feature Gap Analysis](#2-feature-gap-analysis)
3. [Migration Phases](#3-migration-phases)
4. [Phase 1: CLI Compatibility Layer](#4-phase-1-cli-compatibility-layer)
5. [Phase 2: Relocate Vectorized Policies](#5-phase-2-relocate-vectorized-policies)
6. [Phase 3: Update External Imports](#6-phase-3-update-external-imports)
7. [Phase 4: Migrate Tests](#7-phase-4-migrate-tests)
8. [Phase 5: Update Documentation](#8-phase-5-update-documentation)
9. [Phase 6: Deprecation & Removal](#9-phase-6-deprecation--removal)
10. [Rollback Plan](#10-rollback-plan)
11. [Verification Checklist](#11-verification-checklist)

---

## 1. Current State: Import Dependencies

### 1.1 External Files Importing Old Pipeline

The following files **outside** `reinforcement_learning/` import from it and must be updated:

| File | Imports | Migration Action |
|------|---------|------------------|
| `logic/src/models/__init__.py` | Baselines (6 imports) | Use `rl.core.baselines` |
| `logic/src/pipeline/train.py` | epoch, hpo, worker_train | Use `rl/` equivalents |
| `logic/src/policies/neural_agent.py` | local_search | Use `models/policies/vectorized/` |
| `logic/src/utils/logging/visualize_utils.py` | local_search | Use `models/policies/vectorized/` |
| `logic/test/test_models.py` | reinforce_baselines | Use `rl.core.baselines` |
| `logic/test/test_train.py` | epoch, worker_train, trainers | Use Lightning equivalents |
| `logic/test/test_integration.py` | worker_train | Use Lightning trainer |
| `logic/test/test_hp_optim.py` | dehb, hpo | Use `rl.hpo` |
| `logic/test/test_il_train.py` | hgs_vectorized, local_search | Use `models/policies/vectorized/` |
| `logic/test/fixtures/mrl_fixtures.py` | meta strategies | Use `rl.meta` |
| `TUTORIAL.md` | DEHB import example | Update documentation |

### 1.2 Internal Dependencies (Within Old Pipeline)

These are internal to `reinforcement_learning/` and will be removed together:
- `core/` files importing from each other
- `meta/` files importing `weight_strategy`
- `hyperparameter_optimization/` files importing `epoch`, `worker_train`
- `policies/` files importing each other

---

## 2. Feature Gap Analysis

### 2.1 Features Already in New Pipeline (No Action Needed)

| Feature | Old Location | New Location | Status |
|---------|--------------|--------------|--------|
| REINFORCE | `core/reinforce.py` | `rl/core/reinforce.py` | ✅ |
| PPO | `core/ppo.py` | `rl/core/ppo.py` | ✅ |
| SAPO | `core/sapo.py` | `rl/core/sapo.py` | ✅ |
| GSPO | `core/gspo.py` | `rl/core/gspo.py` | ✅ |
| DR-GRPO | `core/dr_grpo.py` | `rl/core/dr_grpo.py` | ✅ |
| NoBaseline | `core/reinforce_baselines.py` | `rl/core/baselines.py` | ✅ |
| ExponentialBaseline | `core/reinforce_baselines.py` | `rl/core/baselines.py` | ✅ |
| RolloutBaseline | `core/reinforce_baselines.py` | `rl/core/baselines.py` | ✅ |
| CriticBaseline | `core/reinforce_baselines.py` | `rl/core/baselines.py` | ✅ |
| WarmupBaseline | `core/reinforce_baselines.py` | `rl/core/baselines.py` | ✅ |
| POMOBaseline | `core/reinforce_baselines.py` | `rl/core/baselines.py` | ✅ |
| RewardWeightOptimizer | `meta/weight_optimizer.py` | `rl/meta/weight_optimizer.py` | ✅ |
| WeightContextualBandit | `meta/contextual_bandits.py` | `rl/meta/contextual_bandits.py` | ✅ |
| MORLWeightOptimizer | `meta/multi_objective.py` | `rl/meta/multi_objective.py` | ✅ |
| CostWeightManager | `meta/temporal_difference_learning.py` | `rl/meta/td_learning.py` | ✅ |
| DEHB | `hyperparameter_optimization/dehb/` | `rl/hpo/dehb.py` | ✅ (simplified) |
| Optuna HPO | `hyperparameter_optimization/hpo.py` | `rl/hpo/optuna_hpo.py` | ✅ |

### 2.2 Features Relocated (Already Done)

| Feature | Old Location | New Location | Status |
|---------|--------------|--------------|--------|
| Vectorized HGS | `policies/hgs_vectorized.py` | `models/policies/vectorized/hgs.py` | ✅ |
| Local Search | `policies/local_search.py` | `models/policies/vectorized/local_search.py` | ✅ |
| Split Algorithm | `policies/split_algorithm.py` | `models/policies/vectorized/split.py` | ✅ |

### 2.3 New Features (Not in Old Pipeline)

| Feature | Location | Description |
|---------|----------|-------------|
| POMO | `rl/core/pomo.py` | Multi-start REINFORCE |
| SymNCO | `rl/core/symnco.py` | Symmetry-aware NCO |
| ImitationLearning | `rl/core/imitation.py` | Expert-guided learning |
| AdaptiveImitation | `rl/core/adaptive_imitation.py` | IL + RL combination |
| HyperNetworkStrategy | `rl/meta/hypernet_strategy.py` | Meta-learning via hypernetworks |
| StateAugmentation | Built into POMO/SymNCO | Dihedral8, symmetric transforms |

### 2.4 Features to Migrate (CLI Arguments)

The old pipeline uses argparse with 150+ arguments. The new pipeline uses Hydra dataclasses. We need a compatibility layer.

**Critical CLI Arguments Not Yet in Hydra Config:**

| Argument | Type | Default | Description | Action |
|----------|------|---------|-------------|--------|
| `--train_time` | bool | False | Multi-day training | Add to TrainConfig |
| `--eval_time_days` | int | 1 | Days for validation | Add to TrainConfig |
| `--area` | str | "riomaior" | County area | Add to EnvConfig |
| `--waste_type` | str | "plastic" | Waste bin type | Add to EnvConfig |
| `--focus_graph` | str | None | Focus graph path | Add to EnvConfig |
| `--focus_size` | int | 0 | Focus graph count | Add to EnvConfig |
| `--temporal_horizon` | int | 0 | TAM history length | Add to ModelConfig |
| `--imitation_weight` | float | 0.0 | IL loss weight | Add to RLConfig |
| `--imitation_decay` | float | 1.0 | IL decay rate | Add to RLConfig |
| `--reannealing_threshold` | float | 0.05 | Reanneal trigger | Add to RLConfig |
| `--post_processing_epochs` | int | 0 | Post-proc epochs | Add to TrainConfig |
| `--wandb_mode` | str | "offline" | W&B mode | Add to root Config |
| `--checkpoint_epochs` | int | 1 | Checkpoint freq | Add to TrainConfig |
| `--shrink_size` | int | None | Batch shrink | Add to TrainConfig |
| `--accumulation_steps` | int | 1 | Gradient accum | Add to TrainConfig |
| `--enable_scaler` | bool | False | Mixed precision | Add to TrainConfig |

**Meta-RL Arguments:**

| Argument | Type | Default | Description | Action |
|----------|------|---------|-------------|--------|
| `--mrl_method` | str | "cb" | Meta-RL method | Already in RLConfig |
| `--mrl_history` | int | 10 | History length | Already in RLConfig |
| `--mrl_exploration_factor` | float | 2.0 | Exploration balance | Add to RLConfig |
| `--hrl_threshold` | float | 0.9 | HRL critical threshold | Add to RLConfig |
| `--hrl_epochs` | int | 4 | HRL PPO epochs | Add to RLConfig |
| `--cb_exploration_method` | str | "ucb" | CB method | Add to RLConfig |
| `--cb_num_configs` | int | 10 | CB configurations | Add to RLConfig |

**HPO Arguments:**

| Argument | Type | Default | Description | Action |
|----------|------|---------|-------------|--------|
| `--hop_method` | str | "dehbo" | HPO method | Already in HPOConfig |
| `--hop_range` | list | [0.0, 2.0] | Search range | Add to HPOConfig |
| `--hop_epochs` | int | 7 | HPO epochs | Add to HPOConfig |
| `--metric` | str | "val_loss" | Optimize metric | Already in HPOConfig |
| `--fevals` | int | 100 | DEHB evaluations | Add to HPOConfig |
| `--n_pop` | int | 20 | DEAP population | Deprecated (DEAP removed) |
| `--n_gen` | int | 10 | DEAP generations | Deprecated (DEAP removed) |

---

## 3. Migration Phases

```
Phase 1: CLI Compatibility Layer                    [Days 1-3]
    └── Add missing Hydra config options
    └── Create argparse → Hydra adapter (optional legacy support)

Phase 2: Relocate Vectorized Policies               [Day 4]
    └── Verify policies in models/policies/vectorized/
    └── Update imports in neural_agent.py, visualize_utils.py

Phase 3: Update External Imports                    [Days 5-7]
    └── Update logic/src/models/__init__.py
    └── Update logic/src/pipeline/train.py
    └── Update logic/src/policies/neural_agent.py
    └── Update logic/src/utils/logging/visualize_utils.py

Phase 4: Migrate Tests                              [Days 8-10]
    └── Update test_models.py, test_train.py, test_integration.py
    └── Update test_hp_optim.py, test_il_train.py
    └── Update fixtures/mrl_fixtures.py
    └── Run full test suite

Phase 5: Update Documentation                       [Days 11-12]
    └── Update TUTORIAL.md
    └── Update CLAUDE.md/AGENTS.md
    └── Update README if needed

Phase 6: Deprecation & Removal                      [Days 13-14]
    └── Add deprecation warnings (optional grace period)
    └── Remove reinforcement_learning/ directory
    └── Final verification
```

---

## 4. Phase 1: CLI Compatibility Layer

### 4.1 Update Hydra Config Dataclasses

**File:** `logic/src/configs/__init__.py`

Add missing fields to existing dataclasses:

```python
@dataclass
class EnvConfig:
    name: str = "vrpp"
    num_loc: int = 50
    min_loc: float = 0.0
    max_loc: float = 1.0
    capacity: Optional[float] = None
    # NEW FIELDS:
    area: str = "riomaior"
    waste_type: str = "plastic"
    focus_graph: Optional[str] = None
    focus_size: int = 0
    eval_focus_size: int = 0
    distance_method: str = "ogd"
    dm_filepath: Optional[str] = None
    waste_filepath: Optional[str] = None
    vertex_method: str = "mmn"
    edge_threshold: float = 0.0
    edge_method: Optional[str] = None


@dataclass
class ModelConfig:
    name: str = "am"
    embed_dim: int = 128
    hidden_dim: int = 512
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_heads: int = 8
    encoder_type: str = "gat"
    # NEW FIELDS:
    temporal_horizon: int = 0
    tanh_clipping: float = 10.0
    normalization: str = "instance"
    activation: str = "gelu"
    dropout: float = 0.1
    mask_inner: bool = True
    mask_logits: bool = True
    mask_graph: bool = False
    spatial_bias: bool = False
    connection_type: str = "residual"


@dataclass
class TrainConfig:
    n_epochs: int = 100
    batch_size: int = 256
    train_data_size: int = 100000
    val_data_size: int = 10000
    val_dataset: Optional[str] = None
    num_workers: int = 4
    # NEW FIELDS:
    train_time: bool = False
    eval_time_days: int = 1
    accumulation_steps: int = 1
    enable_scaler: bool = False
    checkpoint_epochs: int = 1
    shrink_size: Optional[int] = None
    post_processing_epochs: int = 0
    lr_post_processing: float = 0.001
    efficiency_weight: float = 0.8
    overflow_weight: float = 0.2
    log_step: int = 50


@dataclass
class RLConfig:
    algorithm: str = "reinforce"
    baseline: str = "rollout"
    entropy_weight: float = 0.0
    max_grad_norm: float = 1.0
    # PPO family
    ppo_epochs: int = 10
    eps_clip: float = 0.2
    value_loss_weight: float = 0.5
    mini_batch_size: float = 0.25
    # SAPO
    sapo_tau_pos: float = 0.1
    sapo_tau_neg: float = 1.0
    # DR-GRPO
    dr_grpo_group_size: int = 8
    dr_grpo_epsilon: float = 0.2
    # POMO/SymNCO
    num_augment: int = 1
    num_starts: Optional[int] = None
    augment_fn: str = "dihedral8"
    symnco_alpha: float = 0.2
    symnco_beta: float = 1.0
    # Imitation
    expert: str = "hgs"
    imitation_weight: float = 0.0
    imitation_decay: float = 1.0
    imitation_threshold: float = 0.05
    reannealing_threshold: float = 0.05
    reannealing_patience: int = 5
    # Meta-RL
    use_meta: bool = False
    meta_strategy: str = "rnn"  # rnn|bandit|morl|tdl|hypernet
    meta_lr: float = 1e-3
    meta_hidden_dim: int = 64
    meta_history_length: int = 10
    mrl_exploration_factor: float = 2.0
    mrl_range: List[float] = field(default_factory=lambda: [0.01, 5.0])
    # HRL
    hrl_threshold: float = 0.9
    hrl_epochs: int = 4
    hrl_clip_eps: float = 0.2
    # Contextual Bandits
    cb_exploration_method: str = "ucb"
    cb_num_configs: int = 10
    cb_epsilon_decay: float = 0.995
    cb_min_epsilon: float = 0.01


@dataclass
class HPOConfig:
    method: str = "dehbo"
    metric: str = "reward"
    n_trials: int = 20
    n_epochs_per_trial: int = 10
    num_workers: int = 4
    search_space: Dict[str, List[Any]] = field(default_factory=dict)
    # NEW FIELDS:
    hop_range: List[float] = field(default_factory=lambda: [0.0, 2.0])
    fevals: int = 100
    timeout: Optional[int] = None
    n_startup_trials: int = 5
    n_warmup_steps: int = 3
    min_fidelity: int = 1
    max_fidelity: int = 10


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    seed: int = 42
    device: str = "cuda"
    experiment_name: Optional[str] = None
    # NEW FIELDS:
    wandb_mode: str = "offline"
    no_tensorboard: bool = False
    no_progress_bar: bool = False
    output_dir: str = "assets/model_weights"
    log_dir: str = "logs"
```

### 4.2 Optional: Legacy argparse Adapter

Create a function to convert old CLI args to Hydra config:

**File:** `logic/src/cli/legacy_adapter.py`

```python
"""Adapter to convert old argparse arguments to new Hydra config."""

from omegaconf import OmegaConf
from logic.src.configs import Config


def argparse_to_hydra(args) -> Config:
    """Convert argparse namespace to Hydra Config.

    Args:
        args: argparse.Namespace from old CLI

    Returns:
        Config: Hydra configuration object
    """
    cfg = Config()

    # Environment
    cfg.env.name = getattr(args, 'problem', 'vrpp')
    cfg.env.num_loc = getattr(args, 'graph_size', 50)
    cfg.env.area = getattr(args, 'area', 'riomaior')
    cfg.env.waste_type = getattr(args, 'waste_type', 'plastic')
    cfg.env.focus_graph = getattr(args, 'focus_graph', None)
    cfg.env.focus_size = getattr(args, 'focus_size', 0)

    # Model
    cfg.model.name = getattr(args, 'model', 'am')
    cfg.model.embed_dim = getattr(args, 'embedding_dim', 128)
    cfg.model.hidden_dim = getattr(args, 'hidden_dim', 512)
    cfg.model.num_encoder_layers = getattr(args, 'n_encode_layers', 3)
    cfg.model.num_heads = getattr(args, 'n_heads', 8)
    cfg.model.encoder_type = getattr(args, 'encoder', 'gat')
    cfg.model.temporal_horizon = getattr(args, 'temporal_horizon', 0)

    # Training
    cfg.train.n_epochs = getattr(args, 'n_epochs', 100)
    cfg.train.batch_size = getattr(args, 'batch_size', 256)
    cfg.train.train_data_size = getattr(args, 'epoch_size', 100000)
    cfg.train.val_data_size = getattr(args, 'val_size', 10000)
    cfg.train.train_time = getattr(args, 'train_time', False)
    cfg.train.eval_time_days = getattr(args, 'eval_time_days', 1)
    cfg.train.num_workers = getattr(args, 'num_workers', 4)

    # Optimizer
    cfg.optim.optimizer = getattr(args, 'optimizer', 'adam')
    cfg.optim.lr = getattr(args, 'lr_model', 1e-4)

    # RL
    cfg.rl.algorithm = getattr(args, 'rl_algorithm', 'reinforce')
    cfg.rl.baseline = getattr(args, 'baseline', 'rollout') or 'none'
    cfg.rl.entropy_weight = getattr(args, 'entropy_weight', 0.0)
    cfg.rl.max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
    cfg.rl.ppo_epochs = getattr(args, 'ppo_epochs', 10)
    cfg.rl.eps_clip = getattr(args, 'ppo_eps_clip', 0.2)
    cfg.rl.pomo_size = getattr(args, 'pomo_size', 0)
    cfg.rl.imitation_weight = getattr(args, 'imitation_weight', 0.0)
    cfg.rl.imitation_decay = getattr(args, 'imitation_decay', 1.0)

    # Meta-RL
    mrl_method = getattr(args, 'mrl_method', None)
    if mrl_method:
        cfg.rl.use_meta = True
        method_map = {'tdl': 'tdl', 'rwa': 'rnn', 'cb': 'bandit', 'morl': 'morl', 'hrl': 'hrl'}
        cfg.rl.meta_strategy = method_map.get(mrl_method, 'rnn')

    # HPO
    cfg.hpo.method = getattr(args, 'hop_method', 'dehbo')
    cfg.hpo.n_trials = getattr(args, 'n_trials', 20)
    cfg.hpo.metric = getattr(args, 'metric', 'val_loss')
    cfg.hpo.fevals = getattr(args, 'fevals', 100)

    # General
    cfg.seed = getattr(args, 'seed', 42)
    cfg.device = 'cpu' if getattr(args, 'no_cuda', False) else 'cuda'
    cfg.wandb_mode = getattr(args, 'wandb_mode', 'offline')
    cfg.output_dir = getattr(args, 'output_dir', 'assets/model_weights')

    return cfg
```

---

## 5. Phase 2: Relocate Vectorized Policies

### 5.1 Verify Current Location

The vectorized policies have already been relocated to `models/policies/vectorized/`:

```
logic/src/models/policies/vectorized/
├── __init__.py
├── hgs.py           # VectorizedHGS
├── local_search.py  # 2-opt, swap, relocate, etc.
└── split.py         # Linear split algorithm
```

### 5.2 Verify Exports

**File:** `logic/src/models/policies/vectorized/__init__.py`

Ensure all functions are exported:

```python
from logic.src.models.policies.vectorized.local_search import (
    vectorized_two_opt,
    vectorized_swap,
    vectorized_relocate,
    vectorized_two_opt_star,
    vectorized_swap_star,
    vectorized_three_opt,
)
from logic.src.models.policies.vectorized.hgs import (
    VectorizedHGS,
    VectorizedPopulation,
    vectorized_ordered_crossover,
    calc_broken_pairs_distance,
)
from logic.src.models.policies.vectorized.split import (
    vectorized_linear_split,
)
```

---

## 6. Phase 3: Update External Imports

### 6.1 Update `logic/src/models/__init__.py`

**Current:**
```python
from logic.src.pipeline.reinforcement_learning.core.reinforce_baselines import (
    Baseline,
    NoBaseline,
    ExponentialBaseline,
    CriticBaseline,
    RolloutBaseline,
    WarmupBaseline,
)
```

**New:**
```python
from logic.src.pipeline.rl.core.baselines import (
    Baseline,
    NoBaseline,
    ExponentialBaseline,
    CriticBaseline,
    RolloutBaseline,
    WarmupBaseline,
    POMOBaseline,
    get_baseline,
    BASELINE_REGISTRY,
)
```

### 6.2 Update `logic/src/pipeline/train.py`

**Current:**
```python
from logic.src.pipeline.reinforcement_learning.core.epoch import (...)
from logic.src.pipeline.reinforcement_learning.hyperparameter_optimization.hpo import (...)
from logic.src.pipeline.reinforcement_learning.worker_train import (...)
```

**New:**
```python
# Option A: Redirect to Lightning pipeline
from logic.src.pipeline.rl.features.epoch import (
    prepare_epoch,
    regenerate_dataset,
    compute_validation_metrics,
)
from logic.src.pipeline.rl.hpo import OptunaHPO, DifferentialEvolutionHyperband
from logic.src.cli.train_lightning import run_training, run_hpo, create_model

# Option B: Deprecate train.py entirely in favor of train_lightning.py
# Add deprecation warning:
import warnings
warnings.warn(
    "logic.src.pipeline.train is deprecated. Use logic.src.cli.train_lightning instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### 6.3 Update `logic/src/policies/neural_agent.py`

**Current (line 28):**
```python
from logic.src.pipeline.reinforcement_learning.policies.local_search import (
    vectorized_two_opt,
    ...
)
```

**New:**
```python
from logic.src.models.policies.vectorized.local_search import (
    vectorized_two_opt,
    vectorized_swap,
    vectorized_relocate,
    vectorized_two_opt_star,
    vectorized_swap_star,
    vectorized_three_opt,
)
```

### 6.4 Update `logic/src/utils/logging/visualize_utils.py`

**Current (line 27):**
```python
from logic.src.pipeline.reinforcement_learning.policies.local_search import (...)
```

**New:**
```python
from logic.src.models.policies.vectorized.local_search import (
    vectorized_two_opt,
    ...
)
```

---

## 7. Phase 4: Migrate Tests

### 7.1 Update `logic/test/test_models.py`

**Current (line 14):**
```python
from logic.src.pipeline.reinforcement_learning.core.reinforce_baselines import (...)
```

**New:**
```python
from logic.src.pipeline.rl.core.baselines import (
    Baseline,
    NoBaseline,
    ExponentialBaseline,
    CriticBaseline,
    RolloutBaseline,
    WarmupBaseline,
    POMOBaseline,
    get_baseline,
)
```

### 7.2 Update `logic/test/test_train.py`

This file has extensive imports from the old pipeline. Convert tests to use Lightning:

**Current:**
```python
from logic.src.pipeline.reinforcement_learning.core import epoch, post_processing
from logic.src.pipeline.reinforcement_learning.worker_train import (...)
from logic.src.pipeline.reinforcement_learning.core.reinforce import (...)
from logic.src.pipeline.reinforcement_learning.core.ppo import PPOTrainer
# etc.
```

**New:**
```python
import pytorch_lightning as pl
from logic.src.pipeline.rl.core import (
    REINFORCE, PPO, SAPO, GSPO, DRGRPO, POMO, SymNCO,
)
from logic.src.pipeline.rl.core.baselines import get_baseline
from logic.src.pipeline.rl.features.epoch import prepare_epoch
from logic.src.pipeline.trainer import WSTrainer
from logic.src.cli.train_lightning import create_model
```

**Test Migration Example:**

```python
# OLD:
def test_train_epoch():
    from logic.src.pipeline.reinforcement_learning.worker_train import train_reinforce_epoch
    model, _ = train_reinforce_epoch(model, optimizer, baseline, lr_scheduler, ...)

# NEW:
def test_train_epoch():
    from logic.src.cli.train_lightning import create_model
    from logic.src.pipeline.trainer import WSTrainer

    cfg = Config(train=TrainConfig(n_epochs=1))
    model = create_model(cfg)
    trainer = WSTrainer(max_epochs=1, enable_progress_bar=False)
    trainer.fit(model)
```

### 7.3 Update `logic/test/test_integration.py`

Replace worker_train imports with Lightning training:

```python
# OLD:
from logic.src.pipeline.reinforcement_learning.worker_train import (
    train_reinforce_over_time,
    train_reinforce_over_time_cb,
    ...
)

# NEW:
from logic.src.cli.train_lightning import create_model, run_training
from logic.src.configs import Config, RLConfig, TrainConfig
```

### 7.4 Update `logic/test/test_hp_optim.py`

```python
# OLD:
from logic.src.pipeline.reinforcement_learning.hyperparameter_optimization.dehb import (...)
from logic.src.pipeline.reinforcement_learning.hyperparameter_optimization.hpo import (...)

# NEW:
from logic.src.pipeline.rl.hpo import DifferentialEvolutionHyperband, OptunaHPO
from logic.src.cli.train_lightning import run_hpo
```

### 7.5 Update `logic/test/test_il_train.py`

```python
# OLD:
from logic.src.pipeline.reinforcement_learning.policies.hgs_vectorized import (...)
from logic.src.pipeline.reinforcement_learning.policies.local_search import (...)

# NEW:
from logic.src.models.policies.vectorized.hgs import (
    VectorizedHGS, VectorizedPopulation,
)
from logic.src.models.policies.vectorized.local_search import (
    vectorized_two_opt, vectorized_swap, vectorized_relocate,
)
```

### 7.6 Update `logic/test/fixtures/mrl_fixtures.py`

```python
# OLD:
from logic.src.pipeline.reinforcement_learning.meta.contextual_bandits import (...)
from logic.src.pipeline.reinforcement_learning.meta.multi_objective import (...)
from logic.src.pipeline.reinforcement_learning.meta.temporal_difference_learning import (...)
from logic.src.pipeline.reinforcement_learning.meta.weight_optimizer import (...)

# NEW:
from logic.src.pipeline.rl.meta import (
    WeightContextualBandit,
    MORLWeightOptimizer,
    CostWeightManager,
    RewardWeightOptimizer,
    META_STRATEGY_REGISTRY,
)
```

---

## 8. Phase 5: Update Documentation

### 8.1 Update `TUTORIAL.md`

**Line 742 - Update DEHB import:**

```python
# OLD:
from logic.src.pipeline.reinforcement_learning.hyperparameter_optimization.dehb import DEHB

# NEW:
from logic.src.pipeline.rl.hpo import DifferentialEvolutionHyperband
```

### 8.2 Update `CLAUDE.md` / `AGENTS.md`

Update the directory structure section to reflect the new organization:

```markdown
### 10. Pipeline Architecture

```
logic/src/pipeline/
├── rl/                              # NEW: Lightning-based RL pipeline
│   ├── core/                        # RL algorithms & baselines
│   │   ├── base.py                  # RL4COLitModule
│   │   ├── baselines.py             # All baseline implementations
│   │   ├── reinforce.py             # REINFORCE
│   │   ├── ppo.py                   # PPO
│   │   ├── sapo.py                  # SAPO
│   │   ├── gspo.py                  # GSPO
│   │   ├── dr_grpo.py               # DR-GRPO
│   │   ├── pomo.py                  # POMO
│   │   ├── symnco.py                # SymNCO
│   │   ├── imitation.py             # Imitation Learning
│   │   ├── adaptive_imitation.py    # Adaptive IL + RL
│   │   └── hrl.py                   # Hierarchical RL
│   ├── meta/                        # Meta-learning strategies
│   ├── hpo/                         # Hyperparameter optimization
│   └── features/                    # Epoch utilities, post-processing
├── trainer.py                       # WSTrainer (Lightning Trainer)
├── train.py                         # DEPRECATED - use train_lightning.py
└── simulations/                     # Simulation engine
```
```

---

## 9. Phase 6: Deprecation & Removal

### 9.1 Option A: Immediate Removal

If all tests pass after updating imports, remove the old directory:

```bash
# Backup first
cp -r logic/src/pipeline/reinforcement_learning logic/src/pipeline/reinforcement_learning.bak

# Remove
rm -rf logic/src/pipeline/reinforcement_learning

# Run tests to verify
python main.py test_suite
```

### 9.2 Option B: Graceful Deprecation (Recommended)

Add deprecation warnings for a transition period:

**File:** `logic/src/pipeline/reinforcement_learning/__init__.py`

```python
"""
DEPRECATED: This module is deprecated and will be removed in a future version.

Please migrate to the new Lightning-based pipeline:
    - Training: logic.src.cli.train_lightning
    - Algorithms: logic.src.pipeline.rl.core
    - Baselines: logic.src.pipeline.rl.core.baselines
    - Meta-RL: logic.src.pipeline.rl.meta
    - HPO: logic.src.pipeline.rl.hpo
    - Vectorized Policies: logic.src.models.policies.vectorized
"""
import warnings

warnings.warn(
    "logic.src.pipeline.reinforcement_learning is deprecated. "
    "Use logic.src.pipeline.rl and logic.src.cli.train_lightning instead. "
    "This module will be removed in version 4.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backwards compatibility
from logic.src.pipeline.rl.core.baselines import *
from logic.src.pipeline.rl.core import REINFORCE, PPO, SAPO, GSPO, DRGRPO
```

### 9.3 Files to Remove

After migration is complete:

```
logic/src/pipeline/reinforcement_learning/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base.py
│   ├── reinforce.py
│   ├── epoch.py
│   ├── reinforce_baselines.py
│   ├── ppo.py
│   ├── sapo.py
│   ├── gspo.py
│   ├── dr_grpo.py
│   └── post_processing.py
├── meta/
│   ├── __init__.py
│   ├── meta_trainers.py
│   ├── weight_strategy.py
│   ├── weight_optimizer.py
│   ├── contextual_bandits.py
│   ├── temporal_difference_learning.py
│   └── multi_objective.py
├── hyperparameter_optimization/
│   ├── __init__.py
│   ├── hpo.py
│   └── dehb/
├── policies/
│   ├── __init__.py
│   ├── local_search.py
│   ├── hgs_vectorized.py
│   └── split_algorithm.py
├── manager_train.py
└── worker_train.py
```

**Total: ~34 files to remove**

---

## 10. Rollback Plan

If issues are discovered after removal:

1. **Restore from backup:**
   ```bash
   cp -r logic/src/pipeline/reinforcement_learning.bak logic/src/pipeline/reinforcement_learning
   ```

2. **Revert imports:**
   ```bash
   git checkout HEAD~1 -- logic/src/models/__init__.py
   git checkout HEAD~1 -- logic/src/pipeline/train.py
   # etc.
   ```

3. **Re-run tests:**
   ```bash
   python main.py test_suite
   ```

---

## 11. Verification Checklist

### Pre-Migration Verification

- [ ] All tests pass with old pipeline: `python main.py test_suite`
- [ ] Lightning pipeline tests pass: `pytest logic/test/test_rl_lightning.py`
- [ ] Backup created: `reinforcement_learning.bak/`

### Phase 1 Verification

- [ ] Hydra config dataclasses updated
- [ ] `python -c "from logic.src.configs import Config; print(Config())"` works
- [ ] Legacy adapter function works (if created)

### Phase 2 Verification

- [ ] `from logic.src.models.policies.vectorized import *` works
- [ ] All 6 local search functions importable
- [ ] VectorizedHGS importable

### Phase 3 Verification

- [ ] `from logic.src.models import Baseline, NoBaseline, ...` works
- [ ] `logic/src/pipeline/train.py` imports work (or deprecated)
- [ ] `logic/src/policies/neural_agent.py` works
- [ ] `logic/src/utils/logging/visualize_utils.py` works

### Phase 4 Verification

- [ ] `pytest logic/test/test_models.py` passes
- [ ] `pytest logic/test/test_train.py` passes
- [ ] `pytest logic/test/test_integration.py` passes
- [ ] `pytest logic/test/test_hp_optim.py` passes
- [ ] `pytest logic/test/test_il_train.py` passes
- [ ] `pytest logic/test/fixtures/` works

### Phase 5 Verification

- [ ] TUTORIAL.md examples work
- [ ] CLAUDE.md updated
- [ ] No broken links in documentation

### Phase 6 Verification

- [ ] `python main.py test_suite` passes after removal
- [ ] `python main.py train --model am --problem vrpp` works
- [ ] `python main.py hp_optim --hop_method dehbo` works
- [ ] `python main.py mrl_train --mrl_method cb` works
- [ ] No import errors in any module
- [ ] CI/CD pipeline passes

### Post-Migration Verification

- [ ] Remove backup after 1 week of stable operation
- [ ] Update version number in pyproject.toml
- [ ] Tag release in git

---

## Appendix A: Command Reference

### Training Commands (After Migration)

```bash
# Standard training
python logic/src/cli/train_lightning.py \
    +env.name=vrpp \
    +model.name=am \
    +train.n_epochs=100 \
    +rl.algorithm=reinforce \
    +rl.baseline=rollout

# Meta-RL training
python logic/src/cli/train_lightning.py \
    +rl.use_meta=true \
    +rl.meta_strategy=bandit \
    +train.n_epochs=50

# HPO
python logic/src/cli/train_lightning.py \
    +hpo.n_trials=20 \
    +hpo.method=dehbo

# Legacy CLI (if adapter created)
python main.py train \
    --model am \
    --problem vrpp \
    --rl_algorithm ppo \
    --baseline rollout \
    --n_epochs 100
```

### Testing Commands

```bash
# Full test suite
python main.py test_suite

# Specific test modules
pytest logic/test/test_models.py -v
pytest logic/test/test_train.py -v
pytest logic/test/test_integration.py -v

# Lightning-specific tests
pytest logic/test/test_rl_lightning.py -v
```

---

## Appendix B: Import Mapping Reference

| Old Import | New Import |
|------------|------------|
| `reinforcement_learning.core.reinforce_baselines.Baseline` | `rl.core.baselines.Baseline` |
| `reinforcement_learning.core.reinforce_baselines.NoBaseline` | `rl.core.baselines.NoBaseline` |
| `reinforcement_learning.core.reinforce_baselines.ExponentialBaseline` | `rl.core.baselines.ExponentialBaseline` |
| `reinforcement_learning.core.reinforce_baselines.RolloutBaseline` | `rl.core.baselines.RolloutBaseline` |
| `reinforcement_learning.core.reinforce_baselines.CriticBaseline` | `rl.core.baselines.CriticBaseline` |
| `reinforcement_learning.core.reinforce_baselines.WarmupBaseline` | `rl.core.baselines.WarmupBaseline` |
| `reinforcement_learning.core.reinforce_baselines.POMOBaseline` | `rl.core.baselines.POMOBaseline` |
| `reinforcement_learning.core.epoch.rollout` | `rl.features.epoch.compute_validation_metrics` |
| `reinforcement_learning.core.epoch.prepare_epoch` | `rl.features.epoch.prepare_epoch` |
| `reinforcement_learning.core.epoch.validate_update` | `rl.features.epoch.compute_validation_metrics` |
| `reinforcement_learning.worker_train.train_reinforce_epoch` | Use `WSTrainer.fit()` |
| `reinforcement_learning.worker_train.train_reinforce_over_time` | Use `WSTrainer.fit()` with time config |
| `reinforcement_learning.meta.weight_optimizer.RewardWeightOptimizer` | `rl.meta.weight_optimizer.RewardWeightOptimizer` |
| `reinforcement_learning.meta.contextual_bandits.WeightContextualBandit` | `rl.meta.contextual_bandits.WeightContextualBandit` |
| `reinforcement_learning.meta.multi_objective.MORLWeightOptimizer` | `rl.meta.multi_objective.MORLWeightOptimizer` |
| `reinforcement_learning.meta.temporal_difference_learning.CostWeightManager` | `rl.meta.td_learning.CostWeightManager` |
| `reinforcement_learning.hyperparameter_optimization.hpo.*` | `rl.hpo.optuna_hpo.OptunaHPO` |
| `reinforcement_learning.hyperparameter_optimization.dehb.DEHB` | `rl.hpo.dehb.DifferentialEvolutionHyperband` |
| `reinforcement_learning.policies.local_search.*` | `models.policies.vectorized.local_search.*` |
| `reinforcement_learning.policies.hgs_vectorized.*` | `models.policies.vectorized.hgs.*` |
| `reinforcement_learning.policies.split_algorithm.*` | `models.policies.vectorized.split.*` |

---

## Changelog

- **v1.0** (January 21, 2026): Initial migration plan

---

*This document should be followed sequentially. Do not skip phases.*
