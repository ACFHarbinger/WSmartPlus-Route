# Refactoring Plan: Old RL Pipeline → New Lightning Pipeline

> **Version**: 2.0
> **Created**: January 2026
> **Status**: Active
> **Scope**: Integration of `logic/src/pipeline/reinforcement_learning/` features into `logic/src/pipeline/rl/`

---

## Executive Summary

This document details the migration plan for bringing features from the old reinforcement learning pipeline (`logic/src/pipeline/reinforcement_learning/`) into the new PyTorch Lightning-based pipeline (`logic/src/pipeline/rl/`). The new pipeline provides a cleaner, more maintainable architecture but lacks several advanced features from the original implementation.

### Key Statistics

| Aspect | Old Pipeline | New Pipeline | Gap |
|--------|--------------|--------------|-----|
| **Core Files** | 34 files | 11 files | 23 files |
| **RL Algorithms** | 5 (REINFORCE, PPO, SAPO, GSPO, DR-GRPO) | 6 (+ POMO) | ✅ Complete |
| **Baselines** | 6 (No, Exp, POMO, Critic, Rollout, Warmup) | 4 (No, Exp, Rollout, Critic) | 2 missing |
| **Meta-Learning** | 5 strategies + 6 trainers | 1 basic module | Significant gap |
| **HPO** | 6 algorithms (Grid, Random, Bayesian, Hyperband, DEAP, DEHB) | None | Critical gap |
| **Utilities** | Epoch management, time-based training, post-processing | Basic setup/train | Major gap |

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Feature Gap Matrix](#2-feature-gap-matrix)
3. [Implementation Phases](#3-implementation-phases)
4. [Phase 1: Critical Baseline Fixes](#4-phase-1-critical-baseline-fixes)
5. [Phase 2: Epoch Management & Utilities](#5-phase-2-epoch-management--utilities)
6. [Phase 3: Meta-Learning Integration](#6-phase-3-meta-learning-integration)
7. [Phase 4: HPO Integration](#7-phase-4-hpo-integration)
8. [Phase 5: Advanced Features](#8-phase-5-advanced-features)
9. [Migration Strategy](#9-migration-strategy)
10. [Testing Plan](#10-testing-plan)
11. [Risk Assessment](#11-risk-assessment)

---

## 1. Current State Analysis

### 1.1 New Lightning Pipeline (`logic/src/pipeline/rl/`)

```
rl/
├── __init__.py
├── base.py           # RL4COLitModule base class
├── baselines.py      # Baseline implementations (incomplete)
├── reinforce.py      # REINFORCE algorithm
├── ppo.py            # PPO algorithm
├── sapo.py           # SAPO algorithm
├── gspo.py           # GSPO algorithm (incomplete)
├── dr_grpo.py        # DR-GRPO algorithm
├── hrl.py            # HRL module (basic)
├── meta.py           # Meta-RL module (basic)
└── pomo.py           # POMO implementation
```

**Strengths:**
- Clean PyTorch Lightning integration
- Proper separation of concerns
- Hydra configuration support
- TensorDict-based data handling
- Multi-GPU support via Lightning

**Weaknesses:**
- RolloutBaseline returns zeros (placeholder)
- Missing WarmupBaseline
- Missing POMOBaseline in baselines.py
- GSPO missing sequence length normalization
- No dataset regeneration per epoch
- No time-based training support
- No epoch management utilities
- No HPO integration
- Limited meta-learning support

### 1.2 Old Pipeline (`logic/src/pipeline/reinforcement_learning/`)

```
reinforcement_learning/
├── __init__.py
├── manager_train.py              # Manager PPO updates (HRL)
├── worker_train.py               # Worker dispatcher
├── core/
│   ├── base.py                   # BaseReinforceTrainer (Template Method)
│   ├── reinforce.py              # StandardTrainer, TimeTrainer
│   ├── ppo.py                    # PPOTrainer
│   ├── sapo.py                   # SAPOTrainer
│   ├── gspo.py                   # GSPOTrainer
│   ├── dr_grpo.py                # DRGRPOTrainer
│   ├── epoch.py                  # Epoch utilities (critical)
│   ├── reinforce_baselines.py    # Complete baseline implementations
│   └── post_processing.py        # EfficiencyOptimizer
├── meta/
│   ├── weight_strategy.py        # Abstract strategy
│   ├── weight_optimizer.py       # RNN-based meta-learner
│   ├── contextual_bandits.py     # UCB/Thompson Sampling
│   ├── multi_objective.py        # Pareto-based MORL
│   ├── temporal_difference_learning.py  # TD-based weights
│   └── meta_trainers.py          # 6 meta-trainer variants
├── hyperparameter_optimization/
│   ├── hpo.py                    # Main HPO orchestrator
│   └── dehb/                     # DEHB implementation
└── policies/
    ├── hgs_vectorized.py         # GPU-accelerated HGS
    ├── local_search.py           # Vectorized local search
    └── split_algorithm.py        # Split for VRP
```

**Key Features to Port:**
1. Complete RolloutBaseline with T-test updates
2. WarmupBaseline for training stability
3. POMOBaseline implementation
4. Epoch management utilities
5. Time-based training (temporal models)
6. Dataset regeneration per epoch
7. Meta-learning strategies
8. HPO algorithms
9. Post-processing optimization
10. Vectorized policies (optional)

---

## 2. Feature Gap Matrix

### 2.1 Baselines

| Baseline | Old Pipeline | New Pipeline | Action |
|----------|--------------|--------------|--------|
| NoBaseline | ✅ Full | ✅ Full | None |
| ExponentialBaseline | ✅ Full | ✅ Full | None |
| POMOBaseline | ✅ Full | ❌ Missing | **Port** |
| CriticBaseline | ✅ Full | ✅ Full | Done |
| RolloutBaseline | ✅ Full (T-test) | ⚠️ Placeholder | **Critical Fix** |
| WarmupBaseline | ✅ Full | ❌ Missing | **Port** |
| BaselineDataset | ✅ Helper | ❌ Missing | **Port** |

### 2.2 RL Algorithms

| Algorithm | Old Pipeline | New Pipeline | Action |
|-----------|--------------|--------------|--------|
| REINFORCE | ✅ StandardTrainer | ✅ REINFORCE | ✅ Done |
| PPO | ✅ PPOTrainer | ✅ PPO | ✅ Done |
| SAPO | ✅ SAPOTrainer | ✅ SAPO | ✅ Done |
| GSPO | ✅ GSPOTrainer | ✅ GSPO | ✅ Done |
| DR-GRPO | ✅ DRGRPOTrainer | ✅ DRGRPO | ✅ Done |
| POMO | Via POMOBaseline | ✅ POMO module | ✅ Done |
| SymNCO | ❌ Missing | ✅ SymNCO module | ✅ Done |

### 2.3 Training Infrastructure

| Feature | Old Pipeline | New Pipeline | Action |
|---------|--------------|--------------|--------|
| Epoch preparation | ✅ `prepare_epoch()` | ❌ Basic setup | **Port** |
| Batch preparation | ✅ `prepare_batch()` | ❌ Basic | **Port** |
| Validation with metrics | ✅ `validate_update()` | ⚠️ Basic | **Port** |
| Gradient clipping | ✅ `clip_grad_norms()` | ✅ Lightning | Compatible |
| Time-based training | ✅ `TimeTrainer` | ❌ Missing | **Port** |
| Dataset update per day | ✅ `update_time_dataset()` | ❌ Missing | **Port** |
| Fill history (TAM) | ✅ Full support | ❌ Missing | **Port** |
| Checkpoint management | ✅ `complete_train_pass()` | ✅ Lightning | Compatible |

### 2.4 Meta-Learning

| Strategy | Old Pipeline | New Pipeline | Action |
|----------|--------------|--------------|--------|
| RewardWeightOptimizer | ✅ Full (RNN) | ⚠️ Used in meta.py | **Integrate** |
| WeightContextualBandit | ✅ Full | ❌ Missing | **Port** |
| MORLWeightOptimizer | ✅ Full | ❌ Missing | **Port** |
| CostWeightManager (TD) | ✅ Full | ❌ Missing | **Port** |
| WeightAdjustmentStrategy | ✅ Abstract | ❌ Missing | **Port** |
| RWATrainer | ✅ Full | ⚠️ MetaRLModule | **Integrate** |
| ContextualBanditTrainer | ✅ Full | ❌ Missing | **Port** |
| TDLTrainer | ✅ Full | ❌ Missing | **Port** |
| MORLTrainer | ✅ Full | ❌ Missing | **Port** |
| HyperNetworkTrainer | ✅ Full | ❌ Missing | **Port** |
| HRLTrainer | ✅ Full | ⚠️ HRLModule | **Enhance** |

### 2.5 Hyperparameter Optimization

| Algorithm | Old Pipeline | New Pipeline | Action |
|-----------|--------------|--------------|--------|
| Grid Search | ✅ Ray Tune | ❌ Missing | **Port** |
| Random Search | ✅ Ray Tune | ❌ Missing | **Port** |
| Bayesian (Optuna) | ✅ Full | ❌ Missing | **Port** |
| Hyperband | ✅ Ray Tune | ❌ Missing | **Port** |
| DEAP (Genetic) | ✅ Full | ❌ Missing | **Port** |
| DEHB | ✅ Full | ❌ Missing | **Port** |

### 2.6 Utilities & Helpers

| Feature | Old Pipeline | New Pipeline | Action |
|---------|--------------|--------------|--------|
| Post-processing | ✅ EfficiencyOptimizer | ❌ Missing | **Port** |
| decode_routes() | ✅ Full | ❌ Missing | **Port** |
| calculate_efficiency() | ✅ Full | ❌ Missing | **Port** |
| Vectorized HGS | ✅ Full | ❌ Missing | Optional |
| Local Search ops | ✅ Full | ❌ Missing | Optional |
| Split algorithm | ✅ Full | ❌ Missing | Optional |

---

## 3. Implementation Phases

### Overview Timeline

```
Phase 1: Critical Baseline Fixes          [Week 1]
    └── RolloutBaseline, WarmupBaseline, POMOBaseline

Phase 2: Epoch Management & Utilities     [Week 2]
    └── Epoch utils, time-based training, dataset handling

Phase 3: Meta-Learning Integration        [Week 3]
    └── Weight strategies, meta-trainers, HRL enhancement

Phase 4: HPO Integration                  [Week 4]
    └── Optuna, DEHB, Ray Tune integration

Phase 5: Advanced Features                [Week 5+]
    └── Post-processing, vectorized policies (optional)
```

---

## 4. Phase 1: Critical Baseline Fixes

### Priority: HIGH
### Estimated Effort: 2-3 days

### 4.1 Fix RolloutBaseline

**Problem:** Current implementation returns zeros instead of actual greedy rollout values.

**Location:** `logic/src/pipeline/rl/baselines.py:75-82`

**Current (Broken):**
```python
def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
    if self.baseline_policy is None:
        return torch.zeros_like(reward)
    return torch.zeros_like(reward)  # Placeholder
```

**Required Implementation:**
```python
def eval(self, td: TensorDict, reward: torch.Tensor, env=None) -> torch.Tensor:
    """Run greedy baseline rollout."""
    if self.baseline_policy is None:
        return torch.zeros_like(reward)

    with torch.no_grad():
        # Clone td to avoid modifying original
        td_bl = td.clone()
        if env is not None:
            td_bl = env.reset(td_bl)
        out = self.baseline_policy(td_bl, env, decode_type="greedy")
        return out["reward"]

def epoch_callback(self, policy: nn.Module, epoch: int, val_dataset=None, env=None):
    """Update baseline policy if current policy improves significantly."""
    if (epoch + 1) % self.update_every == 0:
        if val_dataset is not None and self.baseline_policy is not None:
            # Evaluate candidate
            candidate_vals = self._rollout(policy, val_dataset, env)
            candidate_mean = candidate_vals.mean().item()

            # Evaluate baseline
            baseline_vals = self._rollout(self.baseline_policy, val_dataset, env)
            baseline_mean = baseline_vals.mean().item()

            # T-test for significance
            from scipy import stats
            t_stat, p_val = stats.ttest_rel(
                candidate_vals.cpu().numpy(),
                baseline_vals.cpu().numpy()
            )

            if candidate_mean > baseline_mean and p_val / 2 < self.bl_alpha:
                self.setup(policy)
        else:
            self.setup(policy)
```

### 4.2 Add WarmupBaseline

**Location:** Add to `logic/src/pipeline/rl/baselines.py`

**Implementation:**
```python
class WarmupBaseline(Baseline):
    """Gradual transition from ExponentialBaseline to target baseline."""

    def __init__(self, baseline: Baseline, n_epochs: int = 1, warmup_beta: float = 0.8):
        self.baseline = baseline
        self.warmup_baseline = ExponentialBaseline(beta=warmup_beta)
        self.alpha = 0.0
        self.n_epochs = n_epochs

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        if self.alpha == 1:
            return self.baseline.eval(td, reward)
        if self.alpha == 0:
            return self.warmup_baseline.eval(td, reward)

        v_target = self.baseline.eval(td, reward)
        v_warmup = self.warmup_baseline.eval(td, reward)
        return self.alpha * v_target + (1 - self.alpha) * v_warmup

    def epoch_callback(self, policy: nn.Module, epoch: int):
        self.baseline.epoch_callback(policy, epoch)
        if epoch < self.n_epochs:
            self.alpha = (epoch + 1) / float(self.n_epochs)
```

### 4.3 Add POMOBaseline

**Location:** Add to `logic/src/pipeline/rl/baselines.py`

**Implementation:**
```python
class POMOBaseline(Baseline):
    """POMO baseline: mean reward across augmentations."""

    def __init__(self, pomo_size: int):
        self.pomo_size = pomo_size

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        # reward: [batch_size * pomo_size]
        B_pomo = reward.size(0)
        B = B_pomo // self.pomo_size

        # Reshape and compute mean
        rewards = reward.view(B, self.pomo_size)
        mean_rewards = rewards.mean(dim=1)

        # Expand back
        return mean_rewards.repeat_interleave(self.pomo_size)
```

### 4.4 Update Baseline Registry

```python
BASELINE_REGISTRY = {
    "none": NoBaseline,
    "exponential": ExponentialBaseline,
    "rollout": RolloutBaseline,
    "critic": CriticBaseline,
    "warmup": WarmupBaseline,
    "pomo": POMOBaseline,
}
```

---

## 5. Phase 2: Epoch Management & Utilities

### Priority: HIGH
### Estimated Effort: 3-4 days

### 5.1 Create Epoch Utilities Module

**New File:** `logic/src/pipeline/rl/utils/epoch.py`

**Contents to Port from** `reinforcement_learning/core/epoch.py`:

1. **`set_decode_type()`** - Set model decoding strategy
2. **`rollout()`** - Complete dataset evaluation
3. **`validate_update()`** - Rich validation with metrics
4. **`clip_grad_norms()`** - Gradient clipping helper
5. **`prepare_batch()`** - Batch preprocessing

### 5.2 Dataset Regeneration

**Modify:** `logic/src/pipeline/rl/base.py`

Add `on_train_epoch_end` enhancement:
```python
def on_train_epoch_end(self):
    """Update baseline and regenerate dataset."""
    if hasattr(self.baseline, "epoch_callback"):
        self.baseline.epoch_callback(self.policy, self.current_epoch)

    # Regenerate training dataset for next epoch
    if self.current_epoch < self.trainer.max_epochs - 1:
        if hasattr(self, 'train_dataset') and hasattr(self.env, 'generator'):
            self.train_dataset = GeneratorDataset(
                self.env.generator,
                self.train_data_size,
            )
```

### 5.3 Time-Based Training Support

**New File:** `logic/src/pipeline/rl/utils/time_training.py`

Port from `reinforcement_learning/core/epoch.py`:
- `prepare_time_dataset()`
- `update_time_dataset()`

Create `TimeBasedMixin` for Lightning modules:
```python
class TimeBasedMixin:
    """Mixin for time-based/temporal training support."""

    def setup_time_training(self, opts):
        """Initialize time-based training state."""
        self.temporal_horizon = opts.get("temporal_horizon", 0)
        self.current_day = 0

    def update_dataset_for_day(self, routes, day):
        """Update dataset state after a day's routes."""
        # Port logic from update_time_dataset()
        pass
```

---

## 6. Phase 3: Meta-Learning Integration

### Priority: MEDIUM
### Estimated Effort: 4-5 days

### 6.1 Port Weight Strategy Interface

**New File:** `logic/src/pipeline/rl/meta/weight_strategy.py`

```python
from abc import ABC, abstractmethod

class WeightAdjustmentStrategy(ABC):
    """Abstract base for meta-learning weight adjustment strategies."""

    @abstractmethod
    def propose_weights(self, context=None) -> dict:
        """Propose new cost weights based on context."""
        pass

    @abstractmethod
    def feedback(self, reward: float, metrics: list, day: int = None):
        """Receive feedback from training."""
        pass

    @abstractmethod
    def get_current_weights(self) -> dict:
        """Get current weight configuration."""
        pass
```

### 6.2 Port Meta-Learning Strategies

**Files to Create:**
- `logic/src/pipeline/rl/meta/__init__.py`
- `logic/src/pipeline/rl/meta/weight_strategy.py`
- `logic/src/pipeline/rl/meta/weight_optimizer.py` (RewardWeightOptimizer)
- `logic/src/pipeline/rl/meta/contextual_bandits.py` (WeightContextualBandit)
- `logic/src/pipeline/rl/meta/multi_objective.py` (MORLWeightOptimizer)
- `logic/src/pipeline/rl/meta/td_learning.py` (CostWeightManager)

### 6.3 Enhance MetaRLModule

**Modify:** `logic/src/pipeline/rl/meta.py`

Add strategy selection:
```python
class MetaRLModule(pl.LightningModule):
    def __init__(
        self,
        agent: Any,
        strategy: str = "rwa",  # rwa, contextual_bandit, morl, tdl
        **kwargs,
    ):
        # ...
        self.meta_strategy = self._create_strategy(strategy, **kwargs)

    def _create_strategy(self, strategy: str, **kwargs):
        if strategy == "rwa":
            return RewardWeightOptimizer(**kwargs)
        elif strategy == "contextual_bandit":
            return WeightContextualBandit(**kwargs)
        elif strategy == "morl":
            return MORLWeightOptimizer(**kwargs)
        elif strategy == "tdl":
            return CostWeightManager(**kwargs)
```

### 6.4 Enhance HRLModule

**Modify:** `logic/src/pipeline/rl/hrl.py`

Add PPO update for manager:
```python
def update_manager_ppo(self, trajectories, gamma=0.99, eps_clip=0.2):
    """PPO update for the manager agent."""
    # Port from manager_train.py
    pass
```

---

## 7. Phase 4: HPO Integration

### Priority: MEDIUM
### Estimated Effort: 5-7 days

### 7.1 Create HPO Module

**New Directory:** `logic/src/pipeline/rl/hpo/`

```
hpo/
├── __init__.py
├── base.py           # Base HPO interface
├── optuna_hpo.py     # Bayesian optimization
├── ray_hpo.py        # Grid/Random/Hyperband
├── dehb/             # DEHB implementation
│   ├── __init__.py
│   ├── dehb.py
│   ├── de.py
│   └── config_space.py
└── utils.py          # Common utilities
```

### 7.2 Lightning-Compatible HPO

Create HPO that works with Lightning training:

```python
class LightningHPO:
    """HPO interface for Lightning-based training."""

    def __init__(self, module_class, env, **base_kwargs):
        self.module_class = module_class
        self.env = env
        self.base_kwargs = base_kwargs

    def optimize_optuna(self, n_trials, search_space, metric="val/reward"):
        """Run Optuna optimization."""
        def objective(trial):
            # Sample hyperparameters
            hparams = {}
            for name, space in search_space.items():
                if space["type"] == "float":
                    hparams[name] = trial.suggest_float(name, space["low"], space["high"])
                elif space["type"] == "int":
                    hparams[name] = trial.suggest_int(name, space["low"], space["high"])

            # Create module and trainer
            module = self.module_class(env=self.env, **self.base_kwargs, **hparams)
            trainer = pl.Trainer(max_epochs=hparams.get("n_epochs", 10))

            # Train and return metric
            trainer.fit(module)
            return trainer.callback_metrics[metric].item()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
```

### 7.3 Port DEHB

**Files to Port:**
- `hyperparameter_optimization/dehb/dehb.py`
- `hyperparameter_optimization/dehb/de.py`
- `hyperparameter_optimization/dehb/de_base.py`
- `hyperparameter_optimization/dehb/dehb_config_repo.py`
- `hyperparameter_optimization/dehb/dehb_shb_manager.py`

---

## 8. Phase 5: Advanced Features

### Priority: LOW
### Estimated Effort: 3-5 days (optional)

### 8.1 Post-Processing Optimization

**New File:** `logic/src/pipeline/rl/utils/post_processing.py`

Port:
- `EfficiencyOptimizer` class
- `calculate_efficiency()` function
- `post_processing_optimization()` function
- `decode_routes()` helper

### 8.2 Vectorized Policies (Optional)

Consider porting if GPU-accelerated heuristics are needed:
- `hgs_vectorized.py` → `logic/src/pipeline/rl/policies/hgs.py`
- `local_search.py` → `logic/src/pipeline/rl/policies/local_search.py`
- `split_algorithm.py` → `logic/src/pipeline/rl/policies/split.py`

---

## 9. Migration Strategy

### 9.1 Approach: Incremental Integration

We recommend **incremental integration** over a full rewrite:

1. **Keep both pipelines operational** during migration
2. **Port features one at a time** with tests
3. **Deprecate old pipeline** only after feature parity
4. **Maintain backwards compatibility** where possible

### 9.2 File Organization

**New Directory Structure:**
```
logic/src/pipeline/rl/
├── __init__.py
├── base.py
├── algorithms/
│   ├── __init__.py
│   ├── reinforce.py
│   ├── ppo.py
│   ├── sapo.py
│   ├── gspo.py
│   ├── dr_grpo.py
│   └── pomo.py
├── baselines.py
├── hrl.py
├── meta/
│   ├── __init__.py
│   ├── weight_strategy.py
│   ├── weight_optimizer.py
│   ├── contextual_bandits.py
│   ├── multi_objective.py
│   ├── td_learning.py
│   └── meta_module.py
├── hpo/
│   ├── __init__.py
│   ├── optuna_hpo.py
│   ├── ray_hpo.py
│   └── dehb/
└── utils/
    ├── __init__.py
    ├── epoch.py
    ├── time_training.py
    └── post_processing.py
```

### 9.3 Import Compatibility

Maintain backward-compatible imports:
```python
# logic/src/pipeline/rl/__init__.py
from .base import RL4COLitModule
from .baselines import (
    Baseline,
    NoBaseline,
    ExponentialBaseline,
    RolloutBaseline,
    CriticBaseline,
    WarmupBaseline,
    POMOBaseline,
    get_baseline,
)
from .algorithms.reinforce import REINFORCE
from .algorithms.ppo import PPO
from .algorithms.sapo import SAPO
from .algorithms.gspo import GSPO
from .algorithms.dr_grpo import DRGRPO
from .algorithms.pomo import POMO
from .hrl import HRLModule
from .meta.meta_module import MetaRLModule
```

---

## 10. Testing Plan

### 10.1 Unit Tests

**New Test Files:**
- `test_baselines.py` - All baseline implementations
- `test_algorithms.py` - RL algorithm correctness
- `test_meta_learning.py` - Meta-learning strategies
- `test_hpo.py` - HPO integration
- `test_time_training.py` - Temporal training

### 10.2 Integration Tests

- Training loop completion
- Checkpoint save/load
- Multi-GPU compatibility
- Hydra configuration

### 10.3 Regression Tests

Compare old vs new pipeline:
```python
def test_parity_reinforce():
    """Ensure new REINFORCE matches old behavior."""
    # Train with old pipeline
    old_result = train_old_reinforce(opts)

    # Train with new pipeline
    new_result = train_new_reinforce(opts)

    # Compare metrics within tolerance
    assert abs(old_result - new_result) < tolerance
```

---

## 11. Risk Assessment

### 11.1 High Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| RolloutBaseline regression | Training instability | Thorough testing, gradual rollout |
| Time-based training bugs | Incorrect temporal learning | Port tests from old pipeline |
| HPO compatibility | Broken hyperparameter search | Isolated module, feature flag |

### 11.2 Medium Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| Meta-learning complexity | Maintenance burden | Clear documentation, tests |
| Performance regression | Slower training | Benchmark before/after |
| API changes | User confusion | Deprecation warnings, migration guide |

### 11.3 Low Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| Import path changes | Minor code updates | Compatibility layer |
| Configuration changes | Config file updates | Hydra defaults |

---

## Appendix A: Code Snippets Reference

### A.1 Old RolloutBaseline (Full Implementation)

```python
# From reinforcement_learning/core/reinforce_baselines.py:436-582
class RolloutBaseline(Baseline):
    def __init__(self, model, problem, opts, epoch=0):
        self.problem = problem
        self.opts = opts
        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        if dataset is None:
            self.dataset = self.problem.make_dataset(...)
        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def epoch_callback(self, model, epoch):
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()
        candidate_mean = candidate_vals.mean()
        if candidate_mean - self.mean < 0:
            t, p = stats.ttest_rel(candidate_vals, self.bl_vals)
            if p / 2 < self.opts["bl_alpha"]:
                self._update_model(model, epoch)
```

### A.2 Old WarmupBaseline (Full Implementation)

```python
# From reinforcement_learning/core/reinforce_baselines.py:132-220
class WarmupBaseline(Baseline):
    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        self.baseline = baseline
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def eval(self, x, c):
        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, loss = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        return (
            self.alpha * v + (1 - self.alpha) * vw,
            self.alpha * loss + (1 - self.alpha) * lw,
        )

    def epoch_callback(self, model, epoch):
        self.baseline.epoch_callback(model, epoch)
        if epoch < self.n_epochs:
            self.alpha = (epoch + 1) / float(self.n_epochs)
```

### A.3 Old POMOBaseline (Full Implementation)

```python
# From reinforcement_learning/core/reinforce_baselines.py:303-354
class POMOBaseline(Baseline):
    def __init__(self, pomo_size):
        self.pomo_size = pomo_size

    def eval(self, x, c):
        # c: [batch_size * pomo_size]
        B_pomo = c.size(0)
        B = B_pomo // self.pomo_size

        # Reshape to [B, pomo_size]
        rewards = c.view(B, self.pomo_size)

        # Compute mean reward per instance: [B]
        mean_rewards = rewards.mean(dim=1)

        # Repeat to match c shape: [B * pomo_size]
        v = mean_rewards.repeat_interleave(self.pomo_size)

        return v, 0  # No critic loss
```

---

## Appendix B: File Mapping

| Old Pipeline File | New Pipeline Location | Status |
|-------------------|----------------------|--------|
| `core/reinforce_baselines.py` | `baselines.py` | Partial |
| `core/epoch.py` | `utils/epoch.py` | To Create |
| `core/reinforce.py` | `reinforce.py` | Complete |
| `core/ppo.py` | `ppo.py` | Complete |
| `core/sapo.py` | `sapo.py` | Complete |
| `core/gspo.py` | `gspo.py` | Needs Fix |
| `core/dr_grpo.py` | `dr_grpo.py` | Complete |
| `core/post_processing.py` | `utils/post_processing.py` | To Create |
| `meta/weight_strategy.py` | `meta/weight_strategy.py` | To Create |
| `meta/weight_optimizer.py` | `meta/weight_optimizer.py` | To Create |
| `meta/contextual_bandits.py` | `meta/contextual_bandits.py` | To Create |
| `meta/multi_objective.py` | `meta/multi_objective.py` | To Create |
| `meta/temporal_difference_learning.py` | `meta/td_learning.py` | To Create |
| `meta/meta_trainers.py` | `meta/meta_module.py` | Partial |
| `hyperparameter_optimization/hpo.py` | `hpo/optuna_hpo.py`, `hpo/ray_hpo.py` | To Create |
| `hyperparameter_optimization/dehb/` | `hpo/dehb/` | To Create |
| `manager_train.py` | `hrl.py` | Partial |
| `worker_train.py` | N/A (Lightning handles) | N/A |
| `policies/hgs_vectorized.py` | `policies/hgs.py` | Optional |
| `policies/local_search.py` | `policies/local_search.py` | Optional |
| `policies/split_algorithm.py` | `policies/split.py` | Optional |

---

## Changelog

- **v2.0** (January 2026): Complete rewrite focusing on old→new pipeline migration
- **v1.0** (Previous): RL4CO vs Logic comparison

---

*This document should be updated as features are ported and tested.*
