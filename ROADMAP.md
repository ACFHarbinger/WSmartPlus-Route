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

> **Version**: 4.0
> **Date**: 2026-02-07
> **Status**: In Progress
> **Sources**: IMPROVEMENT_PLAN.md, REFACTORING_PLAN.md, rl4co v0.6.0 gap analysis

---

## Overview

This document tracks ALL implementation work for WSmart-Route, including the **rl4co parity roadmap**. The original phases (1-7) cover internal improvements. Phases 8-12 address feature gaps against the [rl4co](https://github.com/ai4co/rl4co) library (v0.6.0). Phases 13-14 cover remaining model/component gaps and human-understanding improvements.

Items are marked as:

- ✅ Completed
- 🚧 In Progress
- 📋 Pending

---

## Gap Analysis Summary: WSmart-Route vs rl4co

### Areas Where WSmart-Route is AHEAD

| Capability               | WSmart-Route                                                                                    | rl4co                                            |
| ------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **RL Algorithms**        | 12 (REINFORCE, PPO, A2C, SAPO, GSPO, GDPO, DR-GRPO, POMO, SymNCO, StepwisePPO, IL, Adaptive IL) | 5 (REINFORCE, PPO, StepwisePPO, n-step PPO, A2C) |
| **Meta-Learning**        | MetaRNN, HyperNet, Contextual Bandits, Multi-Objective, TD-Learning, HRL                        | Reptile callback only                            |
| **Classical Solvers**    | Gurobi, HGS, ALNS, PyVRP, OR-Tools, LK, BCP, ACO, SISR, HyperACO, HGS-ALNS                      | PyVRP, OR-Tools, LKH (MTVRP only)                |
| **Selection Strategies** | Regular, LastMinute, LookAhead, Revenue, ServiceLevel, Combined, Manager                        | None                                             |
| **Multi-Day Simulation** | Full event-driven simulator with stochastic bins, checkpointing, 6 distance strategies          | None                                             |
| **Desktop GUI**          | Full PySide6 application with visualization, analysis tabs, file management                     | None                                             |
| **HPO**                  | Optuna + DEHB                                                                                   | Hydra sweeps only                                |
| **Loss Functions**       | NLL, Weighted NLL, KL Divergence, JS Divergence                                                 | Standard REINFORCE only                          |
| **Graph Convolutions**   | Distance-aware, Gated, Efficient multi-head, 5 GCN variants, MPNN, HetGNN                       | Standard GCN + MPNN + GNN + HetGNN               |
| **Operator Libraries**   | Destroy (6), Repair (4), Move (3), Exchange (2), Route (2), Perturbation operators              | None                                             |

### Remaining Parity Gaps (Excluding Environments)

| Category                   | Gap                                     | rl4co Components                                                 | Priority |
| -------------------------- | --------------------------------------- | ---------------------------------------------------------------- | -------- |
| **Constructive AR Models** | 2 missing                               | HAM, L2D                                                         | Medium   |
| **Model Variants**         | 1 missing                               | MVMoE (MoE-enhanced POMO/AM)                                     | Low      |
| **MoE Pointer Attention**  | Not integrated                          | `PointerAttnMoE` (Zhou et al. 2024)                              | Medium   |
| **REINFORCE Baselines**    | 2 missing                               | `SharedBaseline`, `MeanBaseline`                                 | Low      |
| **Meta-Learning**          | 1 missing approach                      | `ReptileCallback` (lightweight meta-learning)                    | Low      |
| **Tensor Operations**      | Missing unified ops module              | `get_distance`, `get_tour_length`, `sparsify_graph`, 12+ more    | Medium   |
| **Edge Embeddings**        | Missing formal edge embedding hierarchy | `TSPEdgeEmbedding`, `CVRPEdgeEmbedding`, `NoEdgeEmbedding`, etc. | Low      |
| **Critic Architecture**    | Less modular                            | `CriticDecoder`, `create_critic_from_actor()`                    | Low      |
| **Code Clarity**           | Several readability concerns            | Duplicate classes, scattered ops, naming inconsistencies         | Medium   |

### Items Previously Listed as Gaps -- Now DONE

| Category                      | Status  | WSmart-Route Implementation                                                             |
| ----------------------------- | ------- | --------------------------------------------------------------------------------------- |
| **Flash Attention**           | ✅ Done | `MultiHeadFlashAttention`, `MultiHeadCrossAttention` using SDPA                         |
| **Data Augmentation**         | ✅ Done | `dihedral_8_augmentation`, `symmetric_augmentation`, `StateAugmentation`                |
| **Structured Evaluation**     | ✅ Done | `GreedyEval`, `SamplingEval`, `AugmentationEval`, `MultiStartEval`, `evaluate_policy()` |
| **Environment Embeddings**    | ✅ Done | Registry-based init/context/dynamic embeddings for all WSmart problem types             |
| **Distribution Utils**        | ✅ Done | `Cluster`, `Mixed`, `Gaussian_Mixture`, `Gamma`, `Empirical`, `Mix_Distribution`        |
| **Positional Embeddings**     | ✅ Done | `AbsolutePositionalEmbedding`, `CyclicPositionalEmbedding`                              |
| **Non-Autoregressive Models** | ✅ Done | `DeepACOPolicy`, `GFACSPolicy`, `NARGNNPolicy` with `NonAutoregressivePolicy` base      |
| **Improvement Models**        | ✅ Done | `DACTPolicy`, `N2SPolicy`, `NeuOptPolicy` with `ImprovementPolicy` base                 |
| **Transductive Models**       | ✅ Done | `TransductiveModel`, `ActiveSearch`, `EAS`, `EASEmb`, `EASLay`                          |
| **MPNN**                      | ✅ Done | `MessagePassingLayer`, `MPNNEncoder`                                                    |
| **Heterogeneous GNN**         | ✅ Done | `HetGNNLayer` with PyG `HeteroConv`                                                     |
| **MatNet**                    | ✅ Done | `MatNetPolicy`, `MatNetEncoder`, `MatNetDecoder`, `MatNetInitEmbedding`                 |
| **MDAM**                      | ✅ Done | `MDAMPolicy`, `MDAMGraphAttentionEncoder`                                               |
| **PolyNet**                   | ✅ Done | `PolyNetPolicy`, `PolyNetAttention`, `PolyNetDecoder`                                   |
| **GLOP**                      | ✅ Done | `GLOPPolicy`, `SubproblemAdapter`                                                       |
| **Constructive AR Models**    | ✅ Done | MatNet, MDAM, PolyNet, GLOP (4 of 6 rl4co AR models)                                    |

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

## Phase 8: Core Infrastructure Alignment ✅

_Foundation work that enables all subsequent model ports._

### 8.1 Flash Attention Integration ✅

**Target**: `logic/src/models/modules/`

- [x] `MultiHeadFlashAttention` using `torch.nn.functional.scaled_dot_product_attention`
- [x] `MultiHeadCrossAttention` with separate Q/KV projections and SDPA
- [x] Backward compatibility with legacy `MultiHeadAttention` (manual matmul)
- [x] `store_attn_weights` fallback mode for visualization hooks

**Files**: `multi_head_flash_attention.py`, `multi_head_cross_attention.py`

---

### 8.2 Modular Environment Embeddings ✅

**Target**: `logic/src/models/embeddings/`

- [x] `INIT_EMBEDDING_REGISTRY` with factory `get_init_embedding(env_name, embed_dim)`
- [x] `CONTEXT_EMBEDDING_REGISTRY` with `EnvContext` base and per-problem subclasses (`VRPPContext`, `CVRPContext`, `WCVRPContext`, `SWCVRPContext`)
- [x] `DYNAMIC_EMBEDDING_REGISTRY` with `StaticEmbedding` and `DynamicEmbedding`
- [x] Init embeddings: `VRPPInitEmbedding`, `CVRPPInitEmbedding`, `WCVRPInitEmbedding`

---

### 8.3 Data Augmentation Module ✅

**Target**: `logic/src/data/transforms.py`

- [x] `dihedral_8_augmentation`, `symmetric_augmentation`, `StateAugmentation`
- [x] Configurable `feats` parameter for selecting which fields to augment

---

### 8.4 Structured Evaluation Pipeline ✅

**Target**: `logic/src/pipeline/eval/`

- [x] `EvalBase` abstract base with `GreedyEval`, `SamplingEval`, `AugmentationEval`, `MultiStartEval`, `MultiStartAugmentEval`
- [x] `evaluate_policy()` dispatcher function
- [x] `get_automatic_batch_size()` for auto-tuning

---

### 8.5 Distribution Utilities ✅

**Target**: `logic/src/data/distributions.py`

- [x] `Cluster`, `Mixed`, `Gaussian_Mixture`, `Gamma`, `Empirical`, `Mix_Distribution`, `Mix_Multi_Distributions`

---

### 8.6 Positional Embeddings ✅

**Target**: `logic/src/models/modules/positional_embeddings.py`

- [x] `AbsolutePositionalEmbedding` (sinusoidal)
- [x] `CyclicPositionalEmbedding` (Ma et al. 2021)

---

## Phase 9: Constructive Autoregressive Models ✅

_Port the remaining constructive autoregressive neural architectures._

### 9.1 MatNet (Matrix Encoding Network) ✅

- [x] `MatNetEncoder`, `MatNetDecoder`, `MatNetPolicy`, `MatNetInitEmbedding`

### 9.2 MDAM (Multi-Decoder Attention Model) ✅

- [x] `MDAMGraphAttentionEncoder`, `MDAMPolicy` with KL divergence

### 9.3 PolyNet ✅

- [x] `PolyNetAttention`, `PolyNetDecoder`, `PolyNetPolicy`

### 9.4 GLOP ✅

- [x] `GLOPPolicy`, `SubproblemAdapter`, `TSPAdapter`, `VRPAdapter`

---

## Phase 10: Non-Autoregressive Models ✅

### 10.1 NAR Policy Base ✅

- [x] `NonAutoregressiveEncoder`, `NonAutoregressiveDecoder`, `NonAutoregressivePolicy`

### 10.2 DeepACO ✅

- [x] `DeepACOPolicy`, `DeepACO` model with ACO integration and 2-opt local search

### 10.3 GFACS ✅

- [x] `GFACSPolicy` with trajectory balance loss

### 10.4 NARGNNPolicy ✅

- [x] `NARGNNEncoder`, `EdgeHeatmapGenerator`, `NARGNNPolicy`

---

## Phase 11: Improvement & Transductive Models ✅

### 11.1 Improvement Model Base ✅

- [x] `ImprovementEncoder`, `ImprovementDecoder`, `ImprovementPolicy`, `ImprovementEnvBase`

### 11.2 DACT ✅

- [x] `DACTEncoder`, `DACTDecoder`, `DACTPolicy`

### 11.3 N2S ✅

- [x] `N2SEncoder`, `N2SDecoder`, `N2SPolicy`

### 11.4 NeuOpt ✅

- [x] `NeuOptEncoder`, `NeuOptDecoder`, `NeuOptPolicy`

### 11.5 Transductive Model Base ✅

- [x] `TransductiveModel` base class with per-instance gradient loop

### 11.6 Active Search ✅

- [x] `ActiveSearch` model

### 11.7 EAS ✅

- [x] `EAS`, `EASEmb`, `EASLay`

---

## Phase 12: Additional NN Components & Graph Modules ✅

_Neural network building blocks that rl4co provides for its model zoo._

### 12.1 Message Passing Neural Network ✅

- [x] `MessagePassingLayer` (PyG-based) and `MPNNEncoder`

### 12.2 Heterogeneous GNN ✅

- [x] `HetGNNLayer` with `HeteroConv` support for bipartite graphs

### 12.3 MoE Pointer Attention ✅

- [x] `PointerAttnMoE` -- MoE-enhanced pointer attention with noisy gating
- [x] Integration with existing MoE modules (`moe.py`, `moe_feed_forward.py`)

**File**: `logic/src/models/modules/pointer_attn_moe.py`

---

### 12.4 Critic Network Enhancements ✅

- [x] Canonical `CriticNetwork` in `logic/src/models/policies/critic.py`
- [x] `create_critic_from_actor()` factory utility
- [x] Legacy `LegacyCriticNetwork` with deprecation warning in `logic/src/models/critic_network.py`
- [x] Old import path re-exports new `CriticNetwork` for backward compatibility
- [x] `A2C` updated to use `LegacyCriticNetwork` explicitly

---

### 12.5 Edge Embeddings ✅

- [x] `EdgeEmbedding` base class
- [x] `TSPEdgeEmbedding` -- distance-based edges with k-NN sparsification
- [x] `CVRPEdgeEmbedding` -- depot-connected sparse graph
- [x] `WCVRPEdgeEmbedding` -- waste collection variant
- [x] `NoEdgeEmbedding` -- dummy edges for node-only problems
- [x] `EDGE_EMBEDDING_REGISTRY` and `get_edge_embedding()` factory

**File**: `logic/src/models/embeddings/edge_embedding.py`

---

## Phase 13: Remaining Model & Baseline Gaps 🚧

_Models and components from rl4co not yet ported._

### 13.1 HAM (Heterogeneous Attention Model) ✅

**Target**: `logic/src/models/policies/ham.py` and `logic/src/models/subnets/encoders/ham_encoder.py`

For pickup-and-delivery problems with heterogeneous node types (pickup vs delivery). Requires cross-attention between node types.

- [x] `PDPEnv` implementation in `logic/src/envs/pdp.py`
- [x] `HeterogeneousAttentionLayer` in `logic/src/models/modules/ham_attention.py`
- [x] `HAMEncoder` in `logic/src/models/subnets/encoders/ham_encoder.py`
- [x] `HAMPolicy` in `logic/src/models/policies/ham.py`
- [x] Tests in `logic/test/unit/models/policies/test_ham.py`

---

- [x] `HeterogeneousAttentionLayer` -- cross-attention between node types (pickup, delivery, depot)
- [x] `GraphHeterogeneousAttentionEncoder` -- multi-layer heterogeneous encoding
- [x] `HeterogeneousAttentionModelPolicy` -- AR policy with type-aware decoding
- [x] `HeterogeneousAttentionModel` (REINFORCE-based)
- [x] Unit tests

**rl4co reference**: `rl4co/models/zoo/ham/` (`attention.py`, `encoder.py`, `policy.py`, `model.py`)

**Dependency**: Requires PDP environment to be implemented.

---

### 13.2 L2D (Learning to Dispatch) ✅

**Target**: `logic/src/models/policies/l2d.py` and `logic/src/models/subnets/{encoders,decoders}/l2d_*.py`

Scheduling-specific model for JSSP/FJSP dispatch decisions. Uses specialized encoders and decoders with scheduling-aware representations.

- [x] `L2DEncoder` -- scheduling graph encoding using heterogeneous message passing
- [x] `L2DDecoder` -- dispatch action decoder (integrated in Policy)
- [x] `L2DPolicy` (REINFORCE)
- [x] `L2DPolicy4PPO` (StepwisePPO variant)
- [x] `L2DModel` (REINFORCE wrapper)
- [x] `L2DPPOModel`
- [x] Unit tests

**rl4co reference**: `rl4co/models/zoo/l2d/` (`encoder.py`, `decoder.py`, `policy.py`, `model.py`)

**Dependency**: Requires scheduling environments (JSSP/FJSP) to be implemented.

---

### 13.3 MVMoE (Multi-View Mixture of Experts) ✅

- [x] `MVMoE_POMO` -- POMO with MoE encoder/decoder kwargs and shared baseline
- [x] `MVMoE_AM` -- AM with MoE encoder/decoder kwargs
- [x] Registered in `RL_ALGORITHM_REGISTRY`

**File**: `logic/src/pipeline/rl/core/mvmoe.py`

---

### 13.4 Additional REINFORCE Baselines ✅

- [x] `MeanBaseline` -- simple batch-mean as baseline value (zero overhead)
- [x] `SharedBaseline` -- shared weight baseline using `create_critic_from_actor`
- [x] Registered in `BASELINE_REGISTRY` as "mean" and "shared"

**File**: `logic/src/pipeline/rl/common/baselines.py`

---

### 13.5 Reptile Meta-Learning Callback ✅

- [x] `ReptileCallback` -- Lightning Callback implementing Reptile (Manchanda et al. 2022)
- [x] Support `data_type` modes: "size", "distribution", "size_distribution"
- [x] Task scheduler with alpha decay for size curriculum
- [x] Configurable task set generation

**File**: `logic/src/pipeline/callbacks.py`

---

## Phase 14: Unified Tensor Operations & Code Clarity 🚧

_Improvements for human understanding, code discoverability, and maintainability._

### 14.1 Unified Tensor Operations Module ✅

**File**: `logic/src/utils/ops.py`

- [x] `get_distance(x, y)` -- Euclidean distance between batched coordinate tensors
- [x] `get_tour_length(ordered_locs)` -- Total tour distance (closed tour)
- [x] `get_open_tour_length(ordered_locs)` -- Total tour distance (open tour)
- [x] `get_distance_matrix(locs)` -- Pairwise Euclidean distance matrix
- [x] `calculate_entropy(logprobs)` -- Entropy of log-probability distributions
- [x] `select_start_nodes(td, num_starts)` -- POMO-style multi-start node selection
- [x] `select_start_nodes_by_distance(td, num_starts)` -- Distance-based start selection
- [x] `get_num_starts(td, env_name)` -- Environment-specific start node count
- [x] `get_best_actions(actions, max_idxs)` -- Select best actions from multi-start rollouts
- [x] `unbatchify_and_gather(x, idx, n)` -- Combined unbatchify + gather operation
- [x] `sparsify_graph(cost_matrix, k_sparse)` -- k-NN graph sparsification for PyG
- [x] `get_full_graph_edge_index(num_node, self_loop)` -- Complete graph edge index (cached)
- [x] `adj_to_pyg_edge_index(adj)` -- Adjacency matrix to PyG edge_index format
- [x] `sample_n_random_actions(td, n)` -- Sample random actions respecting action mask
- [x] `cartesian_to_polar(cartesian, origin)` -- Coordinate system transformation
- [x] `batched_scatter_sum(src, idx)` -- Batched scatter and sum operation
- [x] Re-export existing `batchify`, `unbatchify`, `gather_by_index` from decoding.py

---

### 14.2 CriticNetwork Consolidation ✅

- [x] Canonical implementation in `logic/src/models/policies/critic.py`
- [x] Legacy class renamed to `LegacyCriticNetwork` with deprecation warning
- [x] Old import path (`models/critic_network.py`) re-exports new `CriticNetwork`
- [x] Updated `A2C` to use `LegacyCriticNetwork`

---

### 14.3 Model Registry Pattern ✅

- [x] `_POLICY_REGISTRY_SPEC` mapping model names to policy classes (lazy-loaded)
- [x] `get_policy_class(name)` -- look up policy class by short name
- [x] `get_policy(name, **kwargs)` -- factory function for instantiation
- [x] `RL_ALGORITHM_REGISTRY` mapping algorithm names to Lightning module classes
- [x] `get_rl_algorithm(name)` -- factory function for RL algorithm lookup

**Files**: `logic/src/models/policies/__init__.py`, `logic/src/pipeline/rl/__init__.py`

---

### 14.4 Consistent File Naming 📋

**Target**: `logic/src/models/modules/`

Attention module files use inconsistent naming patterns:

- `multi_head_attention.py` (original manual MHA)
- `multi_head_flash_attention.py` (SDPA-based)
- `multi_head_cross_attention.py` (cross-attention)
- `multi_head_attention_mdam.py` (MDAM-specific)
- `polynet_attention.py` (PolyNet-specific)

- [x] Standardize naming convention: `{variant}_attention.py` for all attention modules
- [x] Update all imports across codebase
- [x] Ensure `__init__.py` re-exports are consistent

---

### 14.5 Module Documentation 📋

**Target**: Package `__init__.py` files

Many `__init__.py` files lack proper module-level docstrings describing what the package contains, making the codebase harder to navigate for newcomers.

- [x] Add descriptive module docstrings to all `__init__.py` files in `logic/src/`
- [x] Add "What's in this package" summaries listing key classes and their purposes
- [x] Add cross-references to related packages (e.g., "See also: logic/src/models/subnets/ for encoder/decoder implementations")

---

### 14.6 Problem-Model Compatibility Matrix 📋

**Target**: Documentation

Create a clear reference showing which models work with which problem types, and which encoders/decoders are compatible.

- [x] Create `COMPATIBILITY.md` with model-problem support matrix
- [x] Document which encoders work with which decoders
- [x] Document which RL algorithms work with which policy types (constructive, improvement, transductive)
- [x] Include recommended configurations per problem type

---

---

## Timeline Summary

| Phase    | Focus                            | Status         |
| -------- | -------------------------------- | -------------- |
| Phase 1  | RL Pipeline Enhancements         | ✅ Completed   |
| Phase 2  | Testing & Quality                | 🚧 In Progress |
| Phase 3  | Documentation                    | 📋 Pending     |
| Phase 4  | Type Safety & Static Analysis    | 📋 Pending     |
| Phase 5  | Code Architecture                | 📋 Pending     |
| Phase 6  | Dependencies & Security          | 📋 Pending     |
| Phase 7  | Developer Tooling                | 📋 Pending     |
| Phase 8  | Core Infrastructure Alignment    | ✅ Completed   |
| Phase 9  | Constructive AR Models           | ✅ Completed   |
| Phase 10 | Non-Autoregressive Models        | ✅ Completed   |
| Phase 11 | Improvement & Transductive       | ✅ Completed   |
| Phase 12 | Additional NN Components         | ✅ Completed   |
| Phase 13 | Remaining Model & Baseline Gaps  | ✅ Completed   |
| Phase 14 | Tensor Operations & Code Clarity | ✅ Completed   |

### Remaining Work

```
Phase 13 (environment-dependent models -- blocked on new envs)
    ├── 13.1 HAM ──────────────> needs PDP environment
    └── 13.2 L2D ──────────────> needs JSSP/FJSP environment

Phase 14 (documentation-only tasks)
    ├── 14.4 File naming ──────> cosmetic, low priority
    ├── 14.5 Module docs ──────> cosmetic, low priority
    └── 14.6 Compatibility matrix > documentation only
```

Phases 3-7 (docs, types, architecture, deps, tooling) can proceed in parallel with any of the parity phases.

---

## Progress Log

### 2026-02-07 (v4.0)

- ✅ Comprehensive re-analysis of WSmart-Route vs rl4co gap (verified every class in codebase)
- ✅ Updated gap analysis: 15+ items previously listed as gaps are now confirmed DONE
- ✅ Identified 9 remaining parity gaps (HAM, L2D, MVMoE, PointerAttnMoE, SharedBaseline, MeanBaseline, ReptileCallback, Edge Embeddings, Critic consolidation)
- ✅ Added Phase 13: Remaining Model & Baseline Gaps
- ✅ Added Phase 14: Unified Tensor Operations & Code Clarity (human understanding improvements)
- ✅ Updated phase statuses: Phase 8 ✅, Phase 9 ✅, Phase 12 🚧
- ✅ Fixed stale execution order diagram
- ✅ Cleaned up duplicate NARGNNPolicy entries in Phase 10.4
- ✅ Fully completed Phase 14: Unified Tensor Operations & Code Clarity
  - Consistent file naming for modules
  - Comprehensive module docstrings
  - Compatibility matrix

### 2026-02-07 (v3.0)

- ✅ Fully completed Phase 10: Non-Autoregressive Models (DeepACO, GFACS, NARGNN)
- ✅ Fully completed Phase 8: Core Infrastructure Alignment (all 6 sub-phases)
- ✅ Phase 9.1-9.4: MatNet, MDAM, PolyNet, GLOP completed
- ✅ Fully completed Phase 11: Improvement & Transductive Models (DACT, N2S, NeuOpt, Active Search, EAS)
- ✅ Consolidated Transductive models into `logic/src/models/policies/common/transductive.py`
- ✅ Phase 12.1-12.2: MPNN, HetGNN completed
- ✅ Phase 5.1: Vectorized ALNS Operators (Destroy, Repair, Move, Exchange) implemented in `logic/src/models/policies/classical/operators/`

### 2026-01-24

- ✅ Created `decoding.py` with full strategy pattern
- ✅ Created `reward_scaler.py` with Welford's algorithm
- ✅ Updated `trainer.py` with RL4CO optimizations
- ✅ Created `a2c.py` with separate actor/critic optimizers
- ✅ Updated `__init__.py` files to export new modules
- ✅ Updated `ConstructivePolicy` to use `DecodingStrategy`

---

## Related Documents

- **[HUMAN_UNDERSTANDING_ROADMAP.md](HUMAN_UNDERSTANDING_ROADMAP.md)** -- Targeted improvements for code readability, documentation coverage, naming consistency, and developer onboarding. Covers docstring gaps, config documentation, complexity reduction, type safety, and naming standards.

---

## Notes

- Decoding module based on RL4CO patterns
- Backward compatible with existing `decode_type` parameter
- New features are opt-in via kwargs
- All baselines (RolloutBaseline, WarmupBaseline, POMOBaseline, etc.) were already complete
- WSmart-Route's domain-specific features (WCVRP, simulation, selection strategies, GUI) are unique and should NOT be removed -- they are strengths beyond rl4co
- rl4co environments and models should be adapted to WSmart-Route's architectural patterns (RL4COEnvBase, factory patterns, Hydra configs)
- Each new environment/model should include: implementation, config files, generators, embeddings, unit tests
- Environment-specific embeddings (TSP, CVRP, OP, PDP, etc.) will be added as their respective environments are implemented
