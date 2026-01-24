# WSmart-Route Implementation Plan

> **Version**: 2.0
> **Date**: 2026-01-24
> **Status**: In Progress
> **Sources**: IMPROVEMENT_PLAN.md, REFACTORING_PLAN.md

---

## Overview

This document tracks ALL implementation work across 7 phases. Items are marked as:
- ✅ Completed
- 🚧 In Progress
- 📋 Pending

---

## Phase 1: RL Pipeline Enhancements (Week 1-2)

### 1.1 Decoding Strategies Module ✅

**File**: `logic/src/utils/functions/decoding.py`

- [x] `DecodingStrategy` abstract base class
- [x] `DecodingConfig` dataclass
- [x] `Greedy` strategy
- [x] `Sampling` strategy with temperature
- [x] `BeamSearch` strategy
- [x] `Evaluate` strategy (for given actions)
- [x] `top_k_filter` utility
- [x] `top_p_filter` utility (nucleus sampling)
- [x] `batchify` / `unbatchify` for multistart
- [x] Tanh clipping (Bello et al.)
- [x] `get_decoding_strategy` factory function

---

### 1.2 RewardScaler ✅

**File**: `logic/src/pipeline/rl/common/reward_scaler.py`

- [x] `RewardScaler` class with Welford's algorithm
- [x] Running mean/variance normalization
- [x] Scale options: "norm", "scale", "none"
- [x] EMA-based alternative
- [x] `BatchRewardScaler` for per-batch normalization

---

### 1.3 Trainer Optimizations ✅

**File**: `logic/src/pipeline/rl/common/trainer.py`

- [x] Auto DDP configuration
- [x] JIT profiling disable
- [x] Matmul precision setting
- [x] Default mixed precision (16-mixed)
- [x] Reload dataloaders every N epochs

---

### 1.4 A2C Algorithm ✅

**File**: `logic/src/pipeline/rl/core/a2c.py`

- [x] Port A2C from RL4CO
- [x] Separate actor/critic optimizers
- [x] Two-optimizer configuration

---

### 1.5 PPO Variants ✅

- [x] N-step PPO (temporal credit assignment)
- [x] Stepwise PPO (per-step updates)

---

### 1.6 Policy Integration ✅
**File**: `logic/src/models/policies/base.py`

- [x] Update `_select_action` to use `DecodingStrategy`
- [x] Add pre/post decoder hooks (implicit in strategy design)
- [x] Add multistart support (via strategy hooks)
- [x] Proper per-step entropy calculation

---

### 1.7 Data Handling ✅

- [x] `ExtraKeyDataset` for baseline rewards
- [x] Multiple named dataloaders (supported by default in Lightning)

---

### 1.8 Utilities ✅

- [x] Optimizer helpers (from strings)
- [x] Scheduler helpers (from strings)
- [x] Lightning device helper

---

## Phase 2: Testing & Quality (Week 3-4)

### 2.1 E2E Tests 📋

- [ ] CLI smoke tests for `train_lightning`
- [ ] CLI smoke tests for `gen_data`
- [ ] CLI smoke tests for `test_sim`
- [ ] CLI smoke tests for `eval`
- [ ] 10+ E2E tests total

---

### 2.2 Integration Tests 📋

- [ ] 30+ integration tests for training workflows
- [ ] 20+ integration tests for simulation workflows
- [ ] Integration tests for evaluation workflows

---

### 2.3 Advanced Testing 📋

- [ ] Property-based tests (hypothesis)
- [ ] Mutation testing (mutmut)
- [ ] Performance benchmarks
- [ ] Contract tests for solver integrations

---

### 2.4 Coverage 📋

- [ ] Coverage badge in README
- [ ] Test coverage ≥ 70%
- [ ] Mutation score ≥ 80%

---

## Phase 3: Documentation (Month 2)

### 3.1 API Documentation 📋

- [ ] Deploy Sphinx to GitHub Pages
- [ ] 100% docstring coverage for public APIs
- [ ] Google-style docstrings enforced
- [ ] Doctest examples in docstrings

---

### 3.2 Developer Experience 📋

- [ ] DEVELOPMENT.md quickstart guide (< 5 min)
- [ ] Tutorial notebook: Training Basics
- [ ] Tutorial notebook: Custom Policies
- [ ] Tutorial notebook: Simulation
- [ ] Architecture diagrams rendered in docs
- [ ] BENCHMARKS.md performance documentation

---

## Phase 4: Type Safety & Static Analysis (Month 2-3)

### 4.1 Type Hints 📋

- [ ] Type-hint top 20 most-imported modules
- [ ] 95% type hint coverage for public APIs
- [ ] Enable strict mypy mode
- [ ] Add type stubs for third-party deps
- [ ] Reduce isinstance checks to < 50
- [ ] Implement protocol classes for duck typing

---

## Phase 5: Code Architecture (Month 3)

### 5.1 Refactoring Large Files 📋

- [ ] Split `solutions.py` (1,518 LOC)
- [ ] Split `hgs_vectorized.py` (1,336 LOC)
- [ ] Ensure no files > 500 LOC

---

### 5.2 Design Patterns 📋

- [ ] Create `interfaces/` module for contracts
- [ ] Break circular dependencies
- [ ] Plugin architecture for policies

---

## Phase 6: Dependencies & Security (Month 3)

### 6.1 Dependency Management 📋

- [ ] Convert exact pins to version ranges
- [ ] Create dependency groups ([gpu], [solvers], [dev], [docs])
- [ ] Document dependency update policy
- [ ] Dependency audit (remove unused)

---

### 6.2 Security 📋

- [ ] Zero known security vulnerabilities (ongoing)

---

## Phase 7: Developer Tooling (Month 4)

### 7.1 Pre-commit & CI 📋

- [ ] Add pre-commit hooks for all checks
- [ ] Complexity checks in CI (radon, mccabe)

---

### 7.2 DevContainer & Docker 📋

- [ ] Create Docker/DevContainer setup

---

## Timeline Summary

| Phase | Duration | Focus | Status |
|-------|----------|-------|--------|
| Phase 1 | Week 1-2 | RL Pipeline | 🚧 In Progress |
| Phase 2 | Week 3-4 | Testing | 📋 Pending |
| Phase 3 | Month 2 | Documentation | 📋 Pending |
| Phase 4 | Month 2-3 | Type Safety | 📋 Pending |
| Phase 5 | Month 3 | Architecture | 📋 Pending |
| Phase 6 | Month 3 | Dependencies | 📋 Pending |
| Phase 7 | Month 4 | Tooling | 📋 Pending |

**Estimated Total**: 4 months

---

## Progress Log

### 2026-01-24
- ✅ Created `decoding.py` with full strategy pattern
- ✅ Created `reward_scaler.py` with Welford's algorithm
- ✅ Updated `trainer.py` with RL4CO optimizations
- ✅ Created `a2c.py` with separate actor/critic optimizers
- ✅ Updated `__init__.py` files to export new modules
- ✅ Updated `ConstructivePolicy` to use `DecodingStrategy`

---

## Notes

- Decoding module based on RL4CO patterns
- Backward compatible with existing `decode_type` parameter
- New features are opt-in via kwargs
- All baselines (RolloutBaseline, WarmupBaseline, POMOBaseline, etc.) were already complete
