# WSmart-Route Lightning Training Pipeline Refactoring Plan

> **Version**: 1.0
> **Date**: January 2026
> **Status**: Draft for Review
> **Purpose**: Comprehensive gap analysis and integration plan for RL4CO features into the new Lightning training pipeline

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Feature Comparison Matrix](#2-feature-comparison-matrix)
3. [Critical Missing Features](#3-critical-missing-features)
4. [Implementation Priorities](#4-implementation-priorities)
5. [Detailed Gap Analysis](#5-detailed-gap-analysis)
6. [Integration Strategies](#6-integration-strategies)
7. [Migration Path](#7-migration-path)
8. [Risk Assessment](#8-risk-assessment)

---

## 1. Executive Summary

This document analyzes the gaps between the **RL4CO library** (`rl4co/rl4co/`) and the **new Lightning training pipeline** (`logic/src/`), identifying missing features and proposing integration strategies.

### Current State

| Aspect | RL4CO | Logic/Src (New Pipeline) |
|--------|-------|--------------------------|
| **Maturity** | Production-ready, well-documented | In-progress integration |
| **RL Algorithms** | 5 (REINFORCE, PPO, A2C, N-step PPO, Stepwise PPO) | 7 (REINFORCE, PPO, SAPO, GSPO, DR-GRPO, HRL, Meta-RL) |
| **Baselines** | 7 types with warmup/statistical testing | 4 types (simplified) |
| **Decoding** | 6 strategies + comprehensive utilities | 2 strategies (basic) |
| **Trainer** | Custom RL4COTrainer with optimizations | Basic WSTrainer wrapper |
| **Config System** | Manual kwargs | Hydra structured configs |

### Key Findings

1. **Logic/Src has unique algorithms** (SAPO, GSPO, DR-GRPO, HRL, Meta-RL) not in RL4CO
2. **RL4CO has production features** missing from Logic/Src (beam search, advanced baselines, reward scaling)
3. **Both share core architecture** but differ in implementation details
4. **Integration opportunity**: Adopt RL4CO utilities while preserving custom algorithms

---

## 2. Feature Comparison Matrix

### 2.1 RL Algorithms

| Algorithm | RL4CO | Logic/Src | Notes |
|-----------|:-----:|:---------:|-------|
| **REINFORCE** | ✅ | ✅ | Both implemented; RL4CO more feature-rich |
| **PPO** | ✅ | ✅ | Both implemented; different value loss functions |
| **A2C** | ✅ | ❌ | Missing - uses CriticBaseline with separate optimizers |
| **N-step PPO** | ✅ | ❌ | Missing - temporal credit assignment |
| **Stepwise PPO** | ✅ | ❌ | Missing - per-step PPO variant |
| **SAPO** | ❌ | ✅ | Custom - Soft Actor-Proxy Optimization |
| **GSPO** | ❌ | ✅ | Custom - Gradient-Scaled Proxy Optimization |
| **DR-GRPO** | ❌ | ✅ | Custom - Divergence-Regularized GRPO |
| **HRL** | ❌ | ✅ | Custom - Manager-Worker architecture |
| **Meta-RL** | Partial | ✅ | Logic/Src has WeightAdjustmentRNN |

### 2.2 Baselines

| Baseline | RL4CO | Logic/Src | Notes |
|----------|:-----:|:---------:|-------|
| **NoBaseline** | ✅ | ✅ (as "none") | Equivalent |
| **ExponentialBaseline** | ✅ | ✅ | Equivalent |
| **MeanBaseline** | ✅ | ❌ | Missing - wrapper around exponential (beta=0) |
| **SharedBaseline** | ✅ | ❌ | Missing - per-batch mean baseline (POMO-style) |
| **CriticBaseline** | ✅ | ✅ | Equivalent |
| **RolloutBaseline** | ✅ | ⚠️ | Logic/Src incomplete (placeholder implementation) |
| **WarmupBaseline** | ✅ | ❌ | Missing - convex combination during warmup |

### 2.3 Decoding Strategies

| Strategy | RL4CO | Logic/Src | Notes |
|----------|:-----:|:---------:|-------|
| **Greedy** | ✅ | ✅ | Basic implementation |
| **Sampling** | ✅ | ✅ | Basic implementation |
| **Multistart Greedy** | ✅ | ❌ | Missing - POMO-style multistart |
| **Multistart Sampling** | ✅ | ❌ | Missing - multi-start sampling |
| **Beam Search** | ✅ | ❌ | Missing - full beam search with backtracking |
| **Evaluate** | ✅ | ❌ | Missing - evaluate given actions |
| **Top-k Filtering** | ✅ | ❌ | Missing - logit filtering |
| **Top-p (Nucleus)** | ✅ | ❌ | Missing - nucleus sampling |
| **Temperature Scaling** | ✅ | ❌ | Missing - temperature parameter |
| **Tanh Clipping** | ✅ | ❌ | Missing - Bello et al. clipping |

### 2.4 Data Handling

| Feature | RL4CO | Logic/Src | Notes |
|---------|:-----:|:---------:|-------|
| **FastTdDataset** | ✅ | ✅ | Equivalent |
| **TensorDictDataset** | ✅ | ✅ | Equivalent |
| **TensorDictDatasetFastGeneration** | ✅ | ✅ | Equivalent |
| **GeneratorDataset** | ✅ | ✅ | Equivalent |
| **ExtraKeyDataset** | ✅ | ❌ | Missing - for baseline rewards |
| **Multiple Dataloaders** | ✅ | ⚠️ | Partial (no named dataloaders) |
| **Dynamic Dataset Regeneration** | ✅ | ⚠️ | Implicit in GeneratorDataset |

### 2.5 Training Infrastructure

| Feature | RL4CO | Logic/Src | Notes |
|---------|:-----:|:---------:|-------|
| **Custom Trainer** | ✅ (RL4COTrainer) | ✅ (WSTrainer) | Different feature sets |
| **Auto DDP Config** | ✅ | ❌ | Missing |
| **JIT Profiling Disable** | ✅ | ❌ | Missing |
| **Matmul Precision** | ✅ | ❌ | Missing |
| **Mixed Precision** | ✅ (default 16-mixed) | ❌ | Not configured |
| **Reload Dataloaders** | ✅ (every epoch) | ❌ | Missing |
| **Gradient Clip Handling** | ✅ (auto for PPO) | ⚠️ | Manual handling |
| **Hydra Config** | ❌ | ✅ | Logic/Src advantage |
| **Optuna HPO** | ❌ | ✅ | Logic/Src advantage |

### 2.6 Utilities

| Utility | RL4CO | Logic/Src | Notes |
|---------|:-----:|:---------:|-------|
| **RewardScaler** | ✅ | ❌ | Missing - running mean/variance normalization |
| **Optimizer Helpers** | ✅ | ❌ | Missing - dynamic creation from strings |
| **Scheduler Helpers** | ✅ | ❌ | Missing - dynamic creation from strings |
| **SpeedMonitor Callback** | ✅ | ✅ | Equivalent (adapted) |
| **Lightning Device Helper** | ✅ | ❌ | Missing |
| **Tensor Operations** | ✅ (batchify, unbatchify, gather) | ❌ | Missing |

### 2.7 Policy Architecture

| Feature | RL4CO | Logic/Src | Notes |
|---------|:-----:|:---------:|-------|
| **ConstructivePolicy Base** | ✅ | ✅ | Different implementations |
| **ImprovementPolicy Base** | ✅ | ✅ | Logic/Src has basic version |
| **Decoding Strategy Integration** | ✅ | ❌ | Missing - policies use inline decoding |
| **Phase-aware Decode Type** | ✅ | ⚠️ | Partial (train vs eval only) |
| **Multi-start Support** | ✅ | ❌ | Missing in policy base |
| **Entropy Calculation** | ✅ | ⚠️ | Missing proper per-step entropy |

---

## 3. Critical Missing Features

### 3.1 HIGH Priority (Required for Parity)

#### 3.1.1 Rollout Baseline (Broken)

**Location**: `logic/src/pipeline/rl/baselines.py:53-88`

**Issue**: The RolloutBaseline implementation is incomplete:
```python
def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
    # ...
    return torch.zeros_like(reward)  # Placeholder
```

**RL4CO Features Missing**:
- Actual greedy rollout evaluation
- Statistical T-test for baseline updates
- Dataset wrapping with baseline rewards (`ExtraKeyDataset`)
- Proper epoch callback with model comparison

**Solution**: Port `rl4co/rl4co/models/rl/reinforce/baselines.py:160-262`

---

#### 3.1.2 Warmup Baseline

**Issue**: No warmup mechanism for baseline transitions.

**RL4CO Implementation**: Convex combination of baselines during warmup epochs:
```python
class WarmupBaseline:
    def eval(self, td, reward, env=None):
        if self.alpha == 1:
            return self.baseline.eval(td, reward, env)
        if self.alpha == 0:
            return self.warmup_baseline.eval(td, reward, env)
        # Convex combination
        return self.alpha * v_b + (1 - self.alpha) * v_wb, ...
```

**Solution**: Add `WarmupBaseline` class that wraps any baseline with exponential warmup.

---

#### 3.1.3 Decoding Strategies Module

**Issue**: Logic/Src has inline, basic decoding in `ConstructivePolicy._select_action()`.

**Missing Capabilities**:
- Temperature scaling
- Top-k / Top-p filtering
- Beam search
- Multistart decoding
- Tanh clipping
- Proper log probability handling

**Solution**: Create `logic/src/utils/decoding.py` based on `rl4co/rl4co/utils/decoding.py`

---

#### 3.1.4 Dataset Regeneration per Epoch

**Issue**: Logic/Src does not regenerate training data each epoch (critical for RL).

**RL4CO Implementation**:
```python
def on_train_epoch_end(self):
    if self.current_epoch < self.trainer.max_epochs - 1:
        self.train_dataset = self.wrap_dataset(
            self.env.dataset(self.data_cfg["train_data_size"], "train")
        )
```

**Solution**: Add `on_train_epoch_end` hook to `RL4COLitModule` base class.

---

### 3.2 MEDIUM Priority (Important Enhancements)

#### 3.2.1 RewardScaler (Advantage Normalization)

**Issue**: Logic/Src has simple mean/std normalization; RL4CO has running statistics.

**RL4CO Implementation**: Welford online algorithm for streaming statistics:
```python
class RewardScaler:
    def __call__(self, scores):
        self.update(scores)
        if self.scale == "norm":
            return (scores - self.mean) / (std + eps)
        elif self.scale == "scale":
            return scores / (std + eps)
```

**Solution**: Port `rl4co/rl4co/models/rl/common/utils.py`

---

#### 3.2.2 RL4COTrainer Optimizations

**Missing Features**:
1. Auto DDP configuration for multi-GPU
2. JIT profiling disable (memory optimization)
3. Matmul precision setting
4. Default mixed precision (16-mixed)
5. Automatic gradient clip handling for manual optimization

**Solution**: Enhance `WSTrainer` with RL4CO optimizations:
```python
class WSTrainer(pl.Trainer):
    def __init__(self, ...):
        # Disable JIT profiling
        torch._C._jit_set_profiling_executor(False)

        # Auto DDP
        if n_devices > 1:
            strategy = DDPStrategy(find_unused_parameters=True)

        # Matmul precision
        torch.set_float32_matmul_precision("medium")
```

---

#### 3.2.3 Shared/Mean Baseline

**Missing**: POMO-style shared baseline (per-batch mean).

**RL4CO Implementation**:
```python
class SharedBaseline(REINFORCEBaseline):
    def eval(self, td, reward, env=None, on_dim=1):
        return reward.mean(dim=on_dim, keepdims=True), 0
```

**Use Case**: Essential for POMO and multistart training.

---

#### 3.2.4 A2C Algorithm

**Missing**: Advantage Actor-Critic with separate optimizer configs.

**RL4CO Implementation**:
- Uses CriticBaseline internally
- Supports separate LR for actor and critic
- Two-optimizer configuration

**Solution**: Port `rl4co/rl4co/models/rl/a2c/a2c.py`

---

### 3.3 LOW Priority (Nice to Have)

#### 3.3.1 PPO Variants

- **N-step PPO**: Temporal credit assignment across steps
- **Stepwise PPO**: Per-step PPO updates

#### 3.3.2 Optimizer/Scheduler Helpers

Dynamic creation from string names (useful for config-driven training).

#### 3.3.3 ExtraKeyDataset

Dataset wrapper that adds baseline rewards for efficient rollout baseline training.

#### 3.3.4 Multistart Decoding

POMO-style multiple starting points with best selection.

---

## 4. Implementation Priorities

### Phase 1: Critical Fixes (Week 1)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Fix RolloutBaseline | Medium | High | `baselines.py` |
| Add dataset regeneration | Low | High | `base.py` |
| Add WarmupBaseline | Low | Medium | `baselines.py` |
| Add SharedBaseline | Low | Medium | `baselines.py` |

### Phase 2: Core Enhancements (Week 2)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Decoding strategies module | High | High | New `decoding.py` |
| RewardScaler | Low | Medium | New `reward_scaler.py` |
| Trainer optimizations | Medium | Medium | `trainer.py` |

### Phase 3: Algorithm Additions (Week 3)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| A2C algorithm | Medium | Medium | New `a2c.py` |
| ExtraKeyDataset | Low | Low | `datasets.py` |
| Multiple dataloaders | Medium | Low | `base.py` |

### Phase 4: Advanced Features (Week 4+)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| N-step PPO | High | Low | New `n_step_ppo.py` |
| Beam search | High | Medium | `decoding.py` |
| Multistart support | Medium | Medium | Policy base classes |

---

## 5. Detailed Gap Analysis

### 5.1 RL4COLitModule Base Class

**RL4CO** (`rl4co/models/rl/common/base.py`):
```python
class RL4COLitModule(LightningModule):
    def __init__(self, ...):
        # Data configuration dictionary
        self.data_cfg = {...}

        # Metric configuration
        self.instantiate_metrics(metrics)

        # Separate batch sizes for train/val/test
        self.val_batch_size, self.test_batch_size

    def setup(self, stage):
        # Dataset setup with wrapping
        self.train_dataset = self.wrap_dataset(...)
        self.setup_loggers()
        self.post_setup_hook()

    def on_train_epoch_end(self):
        # Dataset regeneration
        self.train_dataset = self.wrap_dataset(...)

    def wrap_dataset(self, dataset):
        # Baseline wrapping hook
        return dataset
```

**Logic/Src** (`logic/src/pipeline/rl/base.py`):
```python
class RL4COLitModule(pl.LightningModule, ABC):
    def __init__(self, ...):
        # Simpler parameter storage
        # No data_cfg dictionary
        # Single batch_size

    def setup(self, stage):
        # Basic dataset setup
        # No wrapping support
        # No post_setup_hook

    def on_train_epoch_end(self):
        # Only baseline callback
        # No dataset regeneration
```

**Gaps to Address**:
1. Add `data_cfg` dictionary for configuration
2. Add `wrap_dataset` method for baseline integration
3. Add `post_setup_hook` for subclass extension
4. Add dataset regeneration in `on_train_epoch_end`
5. Add separate val/test batch sizes
6. Add metric configuration system

---

### 5.2 REINFORCE Implementation

**RL4CO** (`rl4co/models/rl/reinforce/reinforce.py`):
```python
class REINFORCE(RL4COLitModule):
    def __init__(self, ..., reward_scale=None):
        self.advantage_scaler = RewardScaler(reward_scale)

    def calculate_loss(self, td, batch, policy_out):
        # Uses batch for extra key (baseline rewards)
        extra = batch.get("extra", None)
        bl_val, bl_loss = self.baseline.eval(...) if extra is None else (extra, 0)
        advantage = self.advantage_scaler(reward - bl_val)

    def post_setup_hook(self):
        self.baseline.setup(self.policy, self.env, ...)

    def wrap_dataset(self, dataset):
        return self.baseline.wrap_dataset(dataset, ...)

    @classmethod
    def load_from_checkpoint(cls, ..., load_baseline=True):
        # Special handling for baseline state dict
```

**Logic/Src** (`logic/src/pipeline/rl/reinforce.py`):
```python
class REINFORCE(RL4COLitModule):
    def calculate_loss(self, td, out, batch_idx):
        baseline_val = self.baseline.eval(td, reward)
        advantage = reward - baseline_val
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
```

**Gaps**:
1. No `RewardScaler` integration
2. No `extra` key support from dataset
3. No `post_setup_hook` call
4. No `wrap_dataset` delegation
5. No custom checkpoint loading for baselines

---

### 5.3 PPO Implementation

**RL4CO** (`rl4co/models/rl/ppo/ppo.py`):
```python
class PPO(RL4COLitModule):
    def __init__(self, ..., normalize_adv=False):
        self.ppo_cfg = {...}  # Config dictionary

    def shared_step(self, batch, batch_idx, phase):
        # Uses env.dataset_cls for mini-batching
        dataset = self.env.dataset_cls(td)

        # Huber loss for value function
        value_loss = F.huber_loss(value_pred, previous_reward)

        # Proper scheduler handling
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if isinstance(sch, MultiStepLR):
            sch.step()
```

**Logic/Src** (`logic/src/pipeline/rl/ppo.py`):
```python
class PPO(RL4COLitModule):
    def __init__(self, ...):
        # Direct attributes instead of config dict

    def training_step(self, batch, batch_idx):
        # Uses FastTdDataset directly
        dataset = FastTdDataset(td)

        # MSE loss for value function
        critic_loss = nn.MSELoss()(values, rewards)

        # No scheduler handling
```

**Gaps**:
1. Use config dictionary for cleaner management
2. Use `env.dataset_cls` for consistency
3. Consider Huber loss for robustness
4. Add scheduler stepping support
5. Add `normalize_adv` option

---

### 5.4 Decoding Strategy Architecture

**RL4CO** (`rl4co/utils/decoding.py`):
```python
class DecodingStrategy(ABC):
    def __init__(self, temperature, top_p, top_k, mask_logits,
                 tanh_clipping, num_samples, multistart, ...):

    def pre_decoder_hook(self, td, env, action):
        # Multistart expansion
        if self.multistart:
            td = batchify(td, self.num_starts)

    def post_decoder_hook(self, td, env):
        # Best selection
        if self.select_best:
            logprobs, actions, td = self._select_best(...)

    def step(self, logits, mask, td, action):
        logprobs = process_logits(logits, mask, temperature, top_p, top_k, ...)

class Greedy(DecodingStrategy): ...
class Sampling(DecodingStrategy): ...
class BeamSearch(DecodingStrategy): ...
class Evaluate(DecodingStrategy): ...

def get_decoding_strategy(name, **config): ...
```

**Logic/Src** (`logic/src/models/policies/base.py`):
```python
class ConstructivePolicy(nn.Module, ABC):
    def _select_action(self, logits, mask, decode_type):
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)

        if decode_type == "greedy":
            action = probs.argmax(dim=-1)
        elif decode_type == "sampling":
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
```

**Gaps**:
1. No strategy pattern (just inline if/else)
2. No temperature scaling
3. No top-k / top-p filtering
4. No multistart support
5. No beam search
6. No pre/post hooks for augmentation

---

## 6. Integration Strategies

### 6.1 Strategy A: Direct Port (Recommended for Utilities)

**Files to port directly**:
- `rl4co/utils/decoding.py` → `logic/src/utils/decoding.py`
- `rl4co/models/rl/common/utils.py` → `logic/src/pipeline/rl/utils.py`
- Baseline implementations

**Advantages**:
- Proven implementations
- Minimal adaptation needed
- Maintains compatibility

---

### 6.2 Strategy B: Adapt and Extend (Recommended for Core Classes)

**For RL4COLitModule**:
1. Keep current Logic/Src structure
2. Add missing methods incrementally
3. Maintain Hydra config compatibility

**For Baselines**:
1. Keep current registry pattern
2. Port individual baseline classes
3. Add `setup()` and `wrap_dataset()` methods

---

### 6.3 Strategy C: Wrapper Pattern (For Complex Features)

**For Decoding**:
1. Create `DecodingStrategy` base class
2. Wrap existing `_select_action` logic
3. Add new strategies progressively

---

## 7. Migration Path

### 7.1 Backward Compatibility

All changes should maintain backward compatibility:
- Default behaviors unchanged
- New features opt-in via config
- Existing experiments reproducible

### 7.2 Testing Strategy

1. **Unit Tests**: Each new component
2. **Integration Tests**: Training pipeline
3. **Regression Tests**: Compare with RL4CO baselines
4. **Performance Tests**: Training speed benchmarks

### 7.3 Documentation

- Update CLAUDE.md with new features
- Add docstrings matching RL4CO style
- Create migration guide for existing code

---

## 8. Risk Assessment

### 8.1 High Risk

| Risk | Mitigation |
|------|------------|
| Breaking existing experiments | Feature flags, extensive testing |
| Performance regression | Benchmark before/after |
| Hydra config conflicts | Careful schema design |

### 8.2 Medium Risk

| Risk | Mitigation |
|------|------------|
| Incomplete port | Phased rollout |
| API inconsistencies | Review against RL4CO |
| Memory issues (beam search) | Add memory guards |

### 8.3 Low Risk

| Risk | Mitigation |
|------|------------|
| Documentation gaps | Parallel documentation |
| Test coverage | Require tests for PRs |

---

## Appendix A: File-by-File Mapping

| RL4CO File | Logic/Src Equivalent | Status |
|------------|---------------------|--------|
| `models/rl/common/base.py` | `pipeline/rl/base.py` | Partial |
| `models/rl/reinforce/reinforce.py` | `pipeline/rl/reinforce.py` | Partial |
| `models/rl/reinforce/baselines.py` | `pipeline/rl/baselines.py` | Partial |
| `models/rl/ppo/ppo.py` | `pipeline/rl/ppo.py` | Partial |
| `models/rl/a2c/a2c.py` | N/A | Missing |
| `models/rl/common/critic.py` | `models/policies/critic.py` | Equivalent |
| `models/rl/common/utils.py` | N/A | Missing |
| `utils/decoding.py` | N/A (inline in policies) | Missing |
| `utils/trainer.py` | `pipeline/trainer.py` | Partial |
| `utils/optim_helpers.py` | N/A | Missing |
| `utils/callbacks/speed_monitor.py` | `callbacks.py` | Equivalent |
| `data/dataset.py` | `data/datasets.py` | Mostly equivalent |
| `envs/common/base.py` | `envs/base.py` | Adapted |

---

## Appendix B: Code Snippets for Quick Reference

### B.1 Rollout Baseline Fix

```python
class RolloutBaseline(Baseline):
    def __init__(self, bl_alpha: float = 0.05, update_every: int = 1):
        self.bl_alpha = bl_alpha
        self.update_every = update_every
        self.policy = None
        self.bl_vals = None

    def setup(self, policy: nn.Module, env, batch_size=64, device="cpu", dataset_size=10000):
        self.policy = copy.deepcopy(policy).to(device)
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False

        # Evaluate on dataset
        dataset = env.dataset(batch_size=[dataset_size])
        self.bl_vals = self._rollout(self.policy, env, dataset, batch_size, device)
        self.mean = self.bl_vals.mean()

    def eval(self, td: TensorDict, reward: torch.Tensor) -> torch.Tensor:
        # For online use, return stored mean
        return torch.full_like(reward, self.mean)

    def epoch_callback(self, policy, env, epoch, **kwargs):
        candidate_vals = self._rollout(policy, env, ...)
        if candidate_vals.mean() > self.mean:
            t, p = ttest_rel(-candidate_vals, -self.bl_vals)
            if p / 2 < self.bl_alpha:
                self.setup(policy, env, **kwargs)
```

### B.2 Dataset Regeneration

```python
def on_train_epoch_end(self):
    # Baseline callback
    if hasattr(self.baseline, "epoch_callback"):
        self.baseline.epoch_callback(self.policy, self.current_epoch)

    # Regenerate training data
    if self.current_epoch < self.trainer.max_epochs - 1:
        self.train_dataset = GeneratorDataset(
            self.env.generator,
            self.train_data_size,
        )
```

### B.3 Decoding Strategy Integration

```python
from logic.src.utils.decoding import get_decoding_strategy

class ConstructivePolicy(nn.Module, ABC):
    def forward(self, td, env, decode_type="sampling", **kwargs):
        strategy = get_decoding_strategy(decode_type, **kwargs)
        td, env, num_starts = strategy.pre_decoder_hook(td, env)

        # ... encoding ...

        while not td["done"].all():
            logits = self.decoder(...)
            td = strategy.step(logits, mask, td)
            td = env.step(td)["next"]

        logprobs, actions, td, env = strategy.post_decoder_hook(td, env)
        return {"reward": ..., "log_likelihood": logprobs.sum(-1), ...}
```

---

**End of Document**
