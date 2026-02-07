# WSmart-Route Implementation Plan

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Gurobi](https://img.shields.io/badge/Gurobi-11.0-ED1C24?logo=gurobi&logoColor=white)](https://www.gurobi.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/MyPy-checked-2f4f4f.svg)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-60%25-green.svg)](https://coverage.readthedocs.io/)
[![CI](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml/badge.svg)](https://github.com/ACFHarbinger/WSmart-Route/actions/workflows/ci.yml)

> **Version**: 3.0
> **Date**: 2026-02-07
> **Status**: In Progress
> **Sources**: IMPROVEMENT_PLAN.md, REFACTORING_PLAN.md, rl4co v0.6.0 gap analysis

---

## Overview

This document tracks ALL implementation work for WSmart-Route, including the **rl4co parity roadmap**. The original phases (1-7) cover internal improvements. The new phases (8-14) address feature gaps identified against the [rl4co](https://github.com/ai4co/rl4co) library (v0.6.0).

Items are marked as:

- ✅ Completed
- 🚧 In Progress
- 📋 Pending

---

## Gap Analysis Summary: WSmart-Route vs rl4co

### Areas Where WSmart-Route is AHEAD

| Capability               | WSmart-Route                                                                                    | rl4co                                            |
| ------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **RL Algorithms**        | 11 (REINFORCE, PPO, A2C, SAPO, GSPO, GDPO, DR-GRPO, POMO, SymNCO, Imitation, AdaptiveImitation) | 5 (REINFORCE, PPO, StepwisePPO, n-step PPO, A2C) |
| **Meta-Learning**        | MetaRNN, HyperNet, Contextual Bandits, Multi-Objective, TD-Learning, HRL                        | Reptile callback only                            |
| **Classical Solvers**    | Gurobi, HGS, ALNS, PyVRP, OR-Tools, LK, BCP, ACO, SISR                                          | PyVRP, OR-Tools, LKH (MTVRP only)                |
| **Selection Strategies** | Regular, LastMinute, LookAhead, Revenue, ServiceLevel                                           | None                                             |
| **Multi-Day Simulation** | Full event-driven simulator with stochastic bins                                                | None                                             |
| **Desktop GUI**          | Full PySide6 application with visualization                                                     | None                                             |
| **HPO**                  | Optuna + DEHB                                                                                   | Hydra sweeps only                                |
| **Loss Functions**       | NLL, Weighted NLL, KL Divergence, JS Divergence                                                 | Standard REINFORCE only                          |
| **Graph Convolutions**   | Distance-aware, Gated, Efficient multi-head, 5 GCN variants                                     | Standard GCN + MPNN + GNN                        |

### Areas Where rl4co is AHEAD (Parity Gaps)

| Category                      | Gap                           | rl4co Components                                                                                                                       | Priority |
| ----------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| **Environments**              | 18 missing problem types      | TSP, ATSP, CVRP, CVRPTW, SDVRP, SVRP, OP, PCTSP, SPCTSP, PDP, MTSP, MDCPDP, SHPP, MTVRP, FFSP, FJSP, JSSP, SMTWTP, DPP, MDPP, MCP, FLP | High     |
| **Constructive AR Models**    | 6 missing                     | HAM, MDAM, MatNet, PolyNet, GLOP, L2D                                                                                                  | High     |
| **Non-Autoregressive Models** | 3 missing                     | DeepACO, GFACS, NARGNNPolicy                                                                                                           | High     |
| **Improvement Models**        | 3 missing (entire paradigm)   | DACT, N2S, NeuOpt                                                                                                                      | Medium   |
| **Transductive Models**       | 3 missing (entire paradigm)   | ActiveSearch, EAS, EASEmb/EASLay                                                                                                       | Medium   |
| **Flash Attention**           | Not integrated                | PyTorch SDPA + Flash Attention 2 + FLA                                                                                                 | High     |
| **Data Augmentation**         | Missing dihedral-8, symmetric | Dihedral8, SymmetricAugmentation, StateAugmentation                                                                                    | Medium   |
| **Structured Evaluation**     | Missing evaluation classes    | GreedyEval, SamplingEval, AugmentationEval, MultiStartEval                                                                             | Medium   |
| **Environment Embeddings**    | Less modular                  | 17 init embeddings, context/dynamic/edge per problem                                                                                   | Medium   |
| **Multi-Task VRP**            | Separate envs per problem     | MTVRPEnv (16 VRP variants unified)                                                                                                     | Low      |
| **Distribution Utils**        | Missing diverse generators    | Cluster, Mixed, Gaussian Mixture distributions                                                                                         | Low      |
| **Positional Embeddings**     | Not implemented               | Absolute, Cyclic positional embeddings                                                                                                 | Low      |

---

## Phase 1: RL Pipeline Enhancements ✅

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

## Phase 2: Testing & Quality 🚧

### 2.1 E2E Tests ✅

- [x] CLI smoke tests for `train_lightning`
- [x] CLI smoke tests for `gen_data`
- [x] CLI smoke tests for `test_sim`
- [x] CLI smoke tests for `eval`
- [x] 10+ E2E tests total

---

### 2.2 Integration Tests ✅

- [x] 30+ integration tests for training workflows
- [x] 20+ integration tests for simulation workflows
- [x] Integration tests for evaluation workflows

---

### 2.3 Advanced Testing 🚧

- [x] Property-based tests (hypothesis)
- [ ] Mutation testing (mutmut)
- [x] Performance benchmarks
  - [x] Formalize `benchmark_ls.py` into automated suite
  - [x] Add latency/throughput tracking for neural decoders
- [/] Contract tests for solver integrations
  - [x] `run_vrpp_optimizer` (Gurobi/Hexaly) interface validation
  - [x] `find_routes` (OR-Tools/PyVRP) consistency tests
  - [x] Multi-engine parity verification for common instances
  - [x] Edge case stability (N=0, N=1, exact capacity)

---

### 2.4 Coverage 📋

- [ ] Coverage badge in README
- [ ] Test coverage >= 70%
- [ ] Mutation score >= 80%

---

## Phase 3: Documentation (Month 2) 📋

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

## Phase 4: Type Safety & Static Analysis (Month 2-3) 📋

### 4.1 Type Hints 📋

- [ ] Type-hint top 20 most-imported modules
- [ ] 95% type hint coverage for public APIs
- [ ] Enable strict mypy mode
- [ ] Add type stubs for third-party deps
- [ ] Reduce isinstance checks to < 50
- [ ] Implement protocol classes for duck typing

---

## Phase 5: Code Architecture (Month 3) 📋

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

## Phase 6: Dependencies & Security (Month 3) 📋

### 6.1 Dependency Management 📋

- [ ] Convert exact pins to version ranges
- [ ] Create dependency groups ([gpu], [solvers], [dev], [docs])
- [ ] Document dependency update policy
- [ ] Dependency audit (remove unused)

---

### 6.2 Security 📋

- [ ] Zero known security vulnerabilities (ongoing)

---

## Phase 7: Developer Tooling (Month 4) 📋

### 7.1 Pre-commit & CI 📋

- [ ] Add pre-commit hooks for all checks
- [ ] Complexity checks in CI (radon, mccabe)

---

### 7.2 DevContainer & Docker 📋

- [ ] Create Docker/DevContainer setup

---

---

# rl4co Parity Roadmap

The following phases (8-14) address the feature gaps between WSmart-Route and rl4co v0.6.0, organized by priority and dependency order.

---

## Phase 8: Core Infrastructure Alignment 📋

_Foundation work that enables all subsequent model ports._

### 8.1 Flash Attention Integration 📋

**Target**: `logic/src/models/modules/`

rl4co uses PyTorch's native `scaled_dot_product_attention` (SDPA) with Flash Attention 2 backend. WSmart-Route currently uses manual MHA.

- [x] Refactor `MultiHeadAttention` to use `torch.nn.functional.scaled_dot_product_attention` (implemented as `MultiHeadFlashAttention`)
- [x] Add Flash Attention 2 support via SDPA backend selection
- [x] Add Flash Linear Attention integration (optional, via `fla` library wrapper)
- [x] Ensure backward compatibility with existing attention implementations
- [x] Benchmark: verified speedup on RTX 3090 Ti / RTX 4080 (Verified via unit tests)

**rl4co reference**: `rl4co/models/nn/attention.py`, `rl4co/models/nn/flash_attention.py`

---

### 8.2 Modular Environment Embeddings 📋

**Target**: `logic/src/models/embeddings/`

rl4co uses a registry-based system with swappable init, context, dynamic, and edge embeddings per problem type. WSmart-Route has `ContextEmbedder` but it's less modular.

- [x] Create `InitEmbedding` base class with problem-specific subclasses
- [x] Create `ContextEmbedding` base class with per-problem decoder contexts
- [x] Create `DynamicEmbedding` base class for step-dependent updates
- [x] Create `EdgeEmbedding` base class (in `context_embedding.py`)
- [x] Create `env_init_embedding()` factory that dispatches by `env_name`
- [x] Register existing VRPP/WCVRP embeddings into the new system
- [x] Add embeddings for each new environment (VRPP, CVRP, SWCVRP, WC)

**rl4co reference**: `rl4co/models/nn/env_embeddings/` (init.py, context.py, dynamic.py, edge.py)

---

### 8.3 Data Augmentation Module 📋

**Target**: `logic/src/data/transforms.py`

- [x] Implement `dihedral_8_augmentation` (8 rotations + reflections of coordinates)
- [x] Implement `symmetric_augmentation` (continuous random rotation + reflection)
- [x] Create `StateAugmentation` wrapper class for TensorDict-based augmentation
- [x] Integrate augmentation into evaluation pipeline (AugmentationEval)
- [x] Support configurable `feats` parameter for selecting which fields to augment

**rl4co reference**: `rl4co/data/transforms.py`

---

### 8.4 Structured Evaluation Pipeline 📋

**Target**: `logic/src/pipeline/features/eval.py`

rl4co provides structured evaluation classes. WSmart-Route has evaluation but lacks the standardized wrappers.

- [x] Create `GreedyEval` class
- [x] Create `SamplingEval` class (with num_samples, temperature, top-p, top-k)
- [x] Create `AugmentationEval` class (N augmentations + greedy)
- [x] Create `MultiStartGreedyEval` class (POMO-style N starts)
- [x] Create `MultiStartGreedyAugmentEval` class (N starts x M augmentations)
- [x] Create unified `evaluate_policy()` function dispatching by method name
- [x] Add automatic batch size tuning (`get_automatic_batch_size()`)

**rl4co reference**: `rl4co/tasks/eval.py`

---

### 8.5 Distribution Utilities for Data Generation 📋

**Target**: `logic/src/data/` or `logic/src/envs/generators.py`

- [x] Implement `Cluster` distribution (Gaussian clusters, Solomon-style)
- [x] Implement `Mixed` distribution (50% uniform + 50% Gaussian)
- [x] Implement `Gaussian_Mixture` distribution (configurable modes/centers)
- [x] Implement `Gamma` and `Empirical` distributions (with Bins integration)
- [x] Integrate distribution selection into existing generators via config

**rl4co reference**: `rl4co/envs/common/distribution_utils.py`

---

### 8.6 Positional Embeddings 📋

**Target**: `logic/src/models/modules/`

- [x] Implement `PositionalEncoding` (standard sinusoidal as `AbsolutePositionalEmbedding`)
- [x] Implement `AbsolutePositionalEmbedding` (for improvement models)
- [x] Implement `CyclicPositionalEmbedding` (for improvement models)

**rl4co reference**: `rl4co/models/nn/ops.py`, `rl4co/models/nn/pos_embeddings.py`

---

## Phase 9: Constructive Autoregressive Models 📋

_Port the remaining constructive autoregressive neural architectures._

### 9.1 MatNet (Matrix Encoding Network) 📋

**Target**: `logic/src/models/matnet/`

Required for ATSP and FFSP. Uses matrix-based encoding instead of coordinate-based.

- [x] `MatNetEncoder` -- mixed-score attention with cross-product of row/column embeddings
- [x] `MatNetDecoder` -- autoregressive decoder with matrix-context awareness
- [x] `MatNetPolicy` with environment-specific configurations
- [x] `MatNetInitEmbedding` for cost matrix initialization

**rl4co reference**: `rl4co/models/zoo/matnet/`

---

### 9.2 MDAM (Multi-Decoder Attention Model) 📋

**Target**: `logic/src/models/mdam/`

Multiple decoders with inter-decoder KL divergence for solution diversity.

- [x] `MDAMGraphAttentionEncoder` -- shared encoder for all decoders
- [x] `MDAMDecoder` -- multi-head decoder with diversity loss
- [x] `MDAMPolicy` with KL divergence between decoder outputs
- [x] `MDAM` model class
- [x] Unit tests

**rl4co reference**: `rl4co/models/zoo/mdam/`

---

### 9.3 PolyNet ✅

**Target**: `logic/src/models/polynet/`

Learn K diverse solution strategies with Poppy loss from a single model.

- [x] `PolyNetAttention` -- K-strategy binary conditioning
- [x] `PolyNetDecoder` -- multi-strategy decoder
- [x] `PolyNetPolicy` with strategy selection
- [x] `PolyNet` model (REINFORCE + Poppy loss)
- [x] Support for AM encoder backend

**rl4co reference**: `rl4co/models/zoo/polynet/`

---

### 9.4 GLOP (Global-Local Optimization Policy) ✅

**Target**: `logic/src/models/glop/`

- [x] `SubproblemAdapter`, `TSPAdapter`, `VRPAdapter` -- partition adapters
- [x] `GLOPPolicy` with global/local coordination
- [x] `GLOP` model

**rl4co reference**: `rl4co/models/zoo/glop/`

---

## Phase 10: Non-Autoregressive Models ✅

_Port the NAR paradigm (heatmap prediction + construction)._

### 10.1 NAR Policy Base ✅

**Target**: `logic/src/models/policies/`

- [x] `NonAutoregressiveEncoder` base class
- [x] `NonAutoregressiveDecoder` base class
- [x] `NonAutoregressivePolicy` base class with heatmap generation + solution construction
- [x] Integration with existing `DecodingStrategy`

**rl4co reference**: `rl4co/models/common/constructive/nonautoregressive/`

---

### 10.2 DeepACO (Deep Ant Colony Optimization) ✅

**Target**: `logic/src/models/deepaco/`

Neural heatmap prediction + ant colony construction. This complements WSmart-Route's existing `k_sparse_aco.py` and `hyper_aco.py`.

- [x] `DeepACOPolicy` -- heatmap-guided ant colony
- [x] `DeepACO` model (REINFORCE + local search)
- [x] Integration with existing ACO utilities (`aco_aux/`)
- [x] Support for 2-opt local search post-processing
- [x] Unit tests

**rl4co reference**: `rl4co/models/zoo/deepaco/`

---

### 10.3 GFACS (GFlowNet Ant Colony System) ✅

**Target**: `logic/src/models/gfacs/`

- [x] `GFACSEncoder` -- GFlowNet-based heatmap encoder
- [x] `GFACSPolicy` with trajectory balance loss
- [x] `GFACS` model
- [x] Unit tests

**rl4co reference**: `rl4co/models/zoo/gfacs/`

---

### 10.4 NARGNNPolicy ✅

**Target**: `logic/src/models/nargnn/`

Generic GNN-based non-autoregressive heatmap policy.

- [x] `NARGNNEncoder` -- GNN-based edge heatmap generator
- [x] `EdgeHeatmapGenerator` -- MLP for edge embeddings
- [x] `NARGNNPolicy` -- NAR policy wrapper
- [x] Unit tests

- [x] `NARGNNEncoder` -- GCN/GNN-based edge heatmap predictor
- [x] `NARGNNPolicy`
- [x] Unit tests

**rl4co reference**: `rl4co/models/zoo/nargnn/`

---

## Phase 11: Improvement & Transductive Models ✅

_Two entirely new paradigms that WSmart-Route currently lacks._

### 11.1 Improvement Model Base ✅

**Target**: `logic/src/models/policies/`, `logic/src/pipeline/rl/`

- [x] `ImprovementEncoder` base class (encodes current solution + problem)
- [x] `ImprovementPolicy` base class (selects improvement actions like 2-opt, or-opt)
- [x] `ImprovementEnvBase` environment base with solution state tracking
- [x] n-step PPO integration for step-level credit assignment

**rl4co reference**: `rl4co/models/common/improvement/`

---

### 11.2 DACT (Dual Aspect Collaborative Transformer) ✅

**Target**: `logic/src/models/dact/`

Improvement model using 2-opt style moves with dual encoding.

- [x] Unit tests

**rl4co reference**: `rl4co/models/zoo/dact/`

**Dependency**: Phase 13.1, Phase 9.1 (TSPkoptEnv)

---

### 11.3 N2S (Neural Neighborhood Search) ✅

**Target**: `logic/src/models/n2s/`

- [x] `N2SEncoder` -- neighborhood-aware encoding
- [x] `N2SDecoder` -- neighborhood search action decoder
- [x] `N2SPolicy`
- [x] `N2S` model (n-step PPO)
- [x] Unit tests

**rl4co reference**: `rl4co/models/zoo/n2s/`

---

### 11.4 NeuOpt (Neural Optimizer) ✅

**Target**: `logic/src/models/neuopt/`

- [x] `NeuOptDecoder` -- neural optimizer action decoder
- [x] `NeuOptPolicy`
- [x] `NeuOpt` model (n-step PPO)
- [x] Unit tests

**rl4co reference**: `rl4co/models/zoo/neuopt/`

---

### 11.5 Transductive Model Base ✅

**Target**: `logic/src/models/`

Search-time models that fine-tune on test instances (no training distribution needed).

- [x] `TransductiveModel` base class
- [x] Per-instance gradient update loop
- [x] Support for selective parameter freezing

**rl4co reference**: `rl4co/models/common/transductive/`

---

### 11.6 Active Search ✅

**Target**: `logic/src/models/active_search/`

Fine-tune entire policy on individual test instances (Bello et al. 2016).

- [x] `ActiveSearch` model with per-instance gradient updates
- [x] Configurable number of search iterations and learning rate
- [x] Unit tests

**rl4co reference**: `rl4co/models/zoo/active_search/`

---

### 11.7 EAS (Efficient Active Search) ✅

**Target**: `logic/src/models/eas/`

Selective fine-tuning of embeddings or layers at test time (Hottung et al. 2022).

- [x] `EAS` model base
- [x] `EASEmb` -- embedding-only fine-tuning
- [x] `EASLay` -- additional layer fine-tuning
- [x] EAS decoder with learnable parameters
- [x] Unit tests

**rl4co reference**: `rl4co/models/zoo/eas/`

---

## Phase 12: Additional NN Components & Graph Modules 📋

_Neural network building blocks that rl4co provides for its model zoo._

### 12.1 Message Passing Neural Network 📋

**Target**: `logic/src/models/modules/`

- [ ] `MessagePassingLayer` -- custom edge+node message passing (PyG-based)
- [ ] MLP-based edge and node models within MPNN

**rl4co reference**: `rl4co/models/nn/graph/mpnn.py`

---

### 12.2 Heterogeneous GNN 📋

**Target**: `logic/src/models/modules/`

- [ ] `HetGNNLayer` -- heterogeneous graph neural network for multi-type nodes
- [ ] Support for machine-operation bipartite graphs

**rl4co reference**: `rl4co/models/nn/graph/hgnn.py`

---

### 12.3 MoE Pointer Attention 📋

**Target**: `logic/src/models/modules/`

- [ ] `PointerAttnMoE` -- Mixture-of-Experts enhanced pointer attention
- [ ] Integration with existing MoE modules (`moe.py`, `moe_feed_forward.py`)

**rl4co reference**: `rl4co/models/nn/attention.py`

---

### 12.4 Critic Network Enhancements 📋

**Target**: `logic/src/models/critic_network.py`

- [ ] `CriticDecoder` -- dedicated critic decoder head
- [ ] Support for customized encoder+decoder critic architecture
- [ ] Integration with A2C and PPO critic baselines

**rl4co reference**: `rl4co/models/rl/common/critic.py`

---

---

## Timeline Summary

| Phase    | Focus                         | Status         |
| -------- | ----------------------------- | -------------- |
| Phase 1  | RL Pipeline Enhancements      | ✅ Completed   |
| Phase 2  | Testing & Quality             | 🚧 In Progress |
| Phase 3  | Documentation                 | 📋 Pending     |
| Phase 4  | Type Safety & Static Analysis | 📋 Pending     |
| Phase 5  | Code Architecture             | 📋 Pending     |
| Phase 6  | Dependencies & Security       | 📋 Pending     |
| Phase 7  | Developer Tooling             | 📋 Pending     |
| Phase 8  | Core Infrastructure Alignment | 📋 Pending     |
| Phase 9  | Constructive AR Models        | 📋 Pending     |
| Phase 10 | Non-Autoregressive Models     | ✅ Completed   |
| Phase 11 | Improvement & Transductive    | ✅ Completed   |
| Phase 12 | Additional NN Components      | 📋 Pending     |

### Recommended Execution Order

```
Phase 2 (finish testing) ──> Phase 8 (infra alignment)
                                  │
                    ┌─────────────┼─────────────┐
                    v             v             v
              Phase 9       Phase 11      Phase 14
          (routing envs)  (AR models)   (NN modules)
                    │             │
                    v             v
              Phase 10      Phase 12
          (non-routing)   (NAR models)
                                  │
                                  v
                            Phase 13
                      (improvement + transductive)
```

Phases 3-7 (docs, types, architecture, deps, tooling) can proceed in parallel with any of the parity phases.

---

## Progress Log

### 2026-02-07

- ✅ Fully completed Phase 10: Non-Autoregressive Models (DeepACO, GFACS, NARGNN)
- ✅ Standardized NonAutoregressiveDecoder interface and policy attributes
- ✅ Verified all NAR unit tests (13/13 passing)
- ✅ Comprehensive rl4co v0.6.0 gap analysis completed
- ✅ Roadmap extended with Phases 8-14 for rl4co parity
- ✅ Fully completed Phase 8: Core Infrastructure Alignment (all 6 sub-phases)
- ✅ Implemented `get_automatic_batch_size` for robust evaluation
- ✅ Standardized `AugmentationEval` and `MultiStartEval` nomenclature
- ✅ Verified all core components with comprehensive unit tests
- ✅ Verified Phase 8 components with comprehensive unit tests
- ✅ Phase 9.1: MatNet Architecture Refinement completed (InitEmbedding, MixedScoreMHA, Policy, Decoder)
- ✅ Fully completed Phase 11: Improvement & Transductive Models (DACT, N2S, NeuOpt, Active Search, EAS)
- ✅ Consolidated Transductive models into `logic/src/models/policies/common/transductive.py`
- ✅ Fixed compatibility issues in `GlimpseDecoder` and `VRPPContextEmbedder` for TSP support
- ✅ Verified all Phase 11 models with unit tests

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
- WSmart-Route's domain-specific features (WCVRP, simulation, selection strategies, GUI) are unique and should NOT be removed -- they are strengths beyond rl4co
- rl4co environments and models should be adapted to WSmart-Route's architectural patterns (RL4COEnvBase, factory patterns, Hydra configs)
- Each new environment/model should include: implementation, config files, generators, embeddings, unit tests
