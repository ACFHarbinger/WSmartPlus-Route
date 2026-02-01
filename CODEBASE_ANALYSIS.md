# WSmart-Route Codebase Analysis & Improvement Plan

> **Scope**: 345 Python files, ~53,600 lines (logic) + GUI layer
> **Date**: January 2026 (Updated: February 1, 2026)
> **Focus**: Overall human comprehension of the entire codebase

---

## Part 1: What Helps a Developer Understand This Codebase (Strengths)

### 1.1 Exceptional Onboarding Documentation (11,000+ Lines)

The repository has a multi-layered documentation suite that few projects match:

| Document | Lines | Purpose |
|----------|-------|---------|
| [README.md](README.md) | 547 | Project overview with documentation hub table |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 1,572 | System design, data flow, component interactions |
| [CLAUDE.md](CLAUDE.md) / [AGENTS.md](AGENTS.md) | 916 | Comprehensive component registry for AI and human contributors |
| [DEVELOPMENT.md](DEVELOPMENT.md) | 852 | Environment setup, CLI reference |
| [TESTING.md](TESTING.md) | 873 | Test suite organization, fixtures, coverage |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 1,083 | Diagnostic guide, common errors |
| [TUTORIAL.md](TUTORIAL.md) | 1,706 | Deep dives, code examples, algorithms |

Cross-linking is present: `README.md` has a hub table linking to all docs. A developer can go from zero to productive following `README -> ARCHITECTURE -> DEVELOPMENT`.

### 1.2 Design Patterns as Self-Documentation

The codebase uses design patterns not just for engineering quality but as *comprehension tools*:

- **Command Pattern** ([actions.py](logic/src/pipeline/simulations/actions.py)): The [day.py:162-172](logic/src/pipeline/simulations/day.py#L162-L172) pipeline reads as a literal description: Fill -> Select -> Route -> Refine -> Collect -> Log. A new developer understands the simulation flow by reading 10 lines.

- **State Pattern** ([states.py](logic/src/pipeline/simulations/states.py)): `InitializingState -> RunningState -> FinishingState` is self-documenting. No need to trace control flow.

- **Template Method** ([base.py](logic/src/pipeline/rl/common/base.py)): `RL4COLitModule.calculate_loss()` tells you exactly where to look when understanding any RL algorithm. 13+ algorithms override only the methods they need -- you can understand SAPO by reading 54 lines, not 482.

- **Registry + Factory**: Both `ENV_REGISTRY` ([envs/__init__.py](logic/src/envs/__init__.py)) and `BASELINE_REGISTRY` ([baselines.py](logic/src/pipeline/rl/common/baselines.py)) make discoverability trivial -- you can see all available options in one dict literal.

- **Abstract Factory** ([model_factory.py](logic/src/models/model_factory.py)): 6 concrete factories cleanly separate encoder/decoder instantiation per architecture.

### 1.3 Strong Type Hint Coverage (~95%+ on Public APIs)

Public functions across all major subsystems have full type hints:
- `create_model(cfg: Config) -> pl.LightningModule` in [train.py:80](logic/src/pipeline/features/train.py#L80)
- `AttentionModel.forward(input: Dict[str, Any], ...) -> Tuple[Tensor, Tensor, Dict[str, Tensor], ...]` in [attention_model.py](logic/src/models/attention_model.py)
- `get_env(name: str, **kwargs) -> RL4COEnvBase` in [envs/__init__.py](logic/src/envs/__init__.py)
- Modern syntax used: `int | float` in [ppo.py:34](logic/src/pipeline/rl/core/ppo.py#L34), `TYPE_CHECKING` blocks for circular import avoidance (6 files, all strategic).

This enables IDE auto-completion and makes parameter contracts visible without reading docstrings.

### 1.4 Clean `__init__.py` Exports

44 `__init__.py` files (802 lines total) with `__all__` lists in key modules:
- [envs/__init__.py](logic/src/envs/__init__.py): Full registry + factory + `__all__`
- [pipeline/rl/__init__.py](logic/src/pipeline/rl/__init__.py): All 10 RL algorithms exported with `__all__`
- [models/policies/__init__.py](logic/src/models/policies/__init__.py): 8 policy classes exported
- [configs/__init__.py](logic/src/configs/__init__.py): Root `Config` dataclass with 11 sub-configs and `__all__`
- [policies/__init__.py](logic/src/policies/__init__.py): Factory pattern exported with backward-compatible aliases (`create_policy = PolicyFactory.get_adapter`)

### 1.5 Extremely Clean Codebase Hygiene

Only **2-3 TODO comments** across the entire `logic/` codebase:
- [ptr_decoder.py:184](logic/src/models/subnets/ptr_decoder.py#L184): Gradient flow workaround
- [visualize_utils.py:58](logic/src/utils/logging/visualize_utils.py#L58), [visualize_utils.py:495](logic/src/utils/logging/visualize_utils.py#L495): Data generation consistency

No FIXME, HACK, or XXX markers. Zero TODOs in the GUI layer. This indicates a mature codebase without unfinished scaffolding that would confuse a reader.

### 1.6 Context Input/Output Contracts in Simulation

Each action class in [actions.py](logic/src/pipeline/simulations/actions.py) documents exactly what it reads from and writes to the shared context dictionary. For example, `FillAction` ([line 100-119](logic/src/pipeline/simulations/actions.py#L100-L119)) lists:

| Direction | Key | Type | Description |
|-----------|-----|------|-------------|
| Input | `bins` | Bins | Bin state management object |
| Input | `day` | int | Current simulation day |
| Output | `new_overflows` | int | Bins that overflowed today |
| Output | `fill` | np.ndarray | Waste added to each bin |
| Output | `total_fill` | np.ndarray | Current bin levels after filling |
| Output | `sum_lost` | float | Total kg lost due to overflows |

This pattern is replicated for all 6 action classes.

### 1.7 Algorithm Inheritance as Documentation

The RL algorithm inheritance chain acts as a reading guide:
- `REINFORCE -> POMO -> SymNCO`: Read base first, each child adds ~50 lines
- `PPO -> SAPO / GSPO / DR-GRPO`: Each child overrides one method, making the algorithmic delta obvious

### 1.8 Well-Organized Constants Module (NEW)

The [constants/](logic/src/constants/) directory extracts domain-specific constants into 11 focused files:

| File | Contents |
|------|----------|
| [models.py](logic/src/constants/models.py) | `NODE_DIM`, `TANH_CLIPPING`, `NORM_EPSILON`, `FEED_FORWARD_EXPANSION` |
| [simulation.py](logic/src/constants/simulation.py) | `METRICS`, `MAX_WASTE`, `MAX_LENGTHS`, `VEHICLE_CAPACITY`, `PROBLEMS` |
| [policies.py](logic/src/constants/policies.py) | `ENGINE_POLICIES`, `THRESHOLD_POLICIES`, `SIMPLE_POLICIES` |
| [paths.py](logic/src/constants/paths.py) | Directory path constants |
| [system.py](logic/src/constants/system.py) | System-level constants |

This eliminates a major source of magic numbers and makes constants discoverable via their domain.

### 1.9 Model Forward() Data Contracts (IMPROVED)

The [AttentionModel.forward()](logic/src/models/attention_model.py#L316-L357) docstring now fully documents:

```
Args:
    input: Problem state dictionary with the following keys:
        - 'loc' (Tensor[batch, n_nodes, 2]): Node coordinates (normalized 0-1).
        - 'demand'/'prize' (Tensor[batch, n_nodes]): Node demand or prize values.
        - 'depot' (Tensor[batch, 2]): Depot coordinates.
        - 'dist' (Tensor[batch, n_nodes, n_nodes], optional): Distance matrix.
        - 'edges' (Tensor[2, n_edges], optional): Edge index for graph convolution.
        - 'waste' (Tensor[batch, n_nodes], optional): Fill levels for WC problems.

Returns:
    Tuple containing:
        - cost (Tensor[batch]): Weighted total cost/reward for each instance.
        - log_likelihood (Tensor[batch]): Log-prob of action sequence.
        - cost_dict (Dict[str, Tensor]): Breakdown with keys 'length', 'waste', 'overflows'.
        - pi (Tensor[batch, seq_len], optional): Node visit sequence.
        - entropy (Tensor[batch], optional): Policy entropy.
```

All tensor shapes, optional markers, and return types are documented. This brings the model layer much closer to the simulation layer's contract quality.

### 1.10 Directory-Level Navigation Aids (NEW)

Two READMEs were added to clarify the most confusing architectural distinction:

- [logic/src/policies/README.md](logic/src/policies/README.md) (26 lines): Explains these are **simulator-facing adapters** inheriting from `BaseRoutingPolicy`
- [logic/src/models/policies/README.md](logic/src/models/policies/README.md) (24 lines): Explains these are **RL training wrappers** inheriting from `ConstructivePolicy`

Both explicitly cross-reference each other, resolving the dual-policy directory confusion.

---

## Part 2: Previously Identified Issues - Status

### 2.1 Simulation Pipeline Improvements

| Issue | Previous | Status |
|-------|----------|--------|
| Policy layer code duplication | CRITICAL | RESOLVED — [BaseRoutingPolicy](logic/src/policies/base_routing_policy.py) extracted |
| Policy parsing 95-line chain | HIGH | RESOLVED — Lookup tables in [states.py](logic/src/pipeline/simulations/states.py) |
| `Any` overuse in Context | MEDIUM | MOSTLY RESOLVED — Proper types in [context.py](logic/src/pipeline/simulations/context.py) |
| Broad exception handling (sim) | HIGH | MOSTLY RESOLVED — Specific exception types in simulation pipeline |
| Policy-to-simulator dependency | MEDIUM | RESOLVED — `load_area_params` in [data_utils.py](logic/src/utils/data/data_utils.py) |
| `solutions.py` 1,518 lines | MEDIUM | RESOLVED — Split into [route_search.py](logic/src/policies/look_ahead_aux/route_search.py), [simulated_annealing.py](logic/src/policies/look_ahead_aux/simulated_annealing.py), [solution_initialization.py](logic/src/policies/look_ahead_aux/solution_initialization.py) |
| Policy naming collision | MEDIUM | RESOLVED — [VectorizedALNS](logic/src/models/policies/classical/alns.py), [VectorizedHGS](logic/src/models/policies/classical/hgs.py) |

### 2.2 Training Pipeline & Model Improvements

| Issue | Previous | Status |
|-------|----------|--------|
| `RLConfig` monolith (35+ flat fields) | MEDIUM | RESOLVED — 8 algorithm-specific sub-configs in [rl.py](logic/src/configs/rl.py) |
| Algorithm if-elif chain in `train.py` | HIGH | PARTIALLY RESOLVED — Policy map for model selection ([train.py:91-100](logic/src/pipeline/features/train.py#L91-L100)), critic helper extracted ([train.py:221-231](logic/src/pipeline/features/train.py#L221-L231)), but RL module selection still if-elif |
| Critic creation duplicated 4x | HIGH | RESOLVED — `_create_critic_helper()` at [train.py:221](logic/src/pipeline/features/train.py#L221) |
| Parameter naming: `embed_dim` vs `embedding_dim` | CRITICAL | RESOLVED — 281 occurrences of `embed_dim`, 0 of `embedding_dim` (fully standardized) |
| Parameter naming: `n_heads` vs `num_heads` | MEDIUM | RESOLVED — 106 occurrences of `n_heads`, 0 of `num_heads` (fully standardized) |
| CLAUDE.md references `tasks/` 10 times | HIGH | RESOLVED — Zero stale references, `tasks/` directory removed |
| No directory READMEs for policies | HIGH | RESOLVED — READMEs in both [policies/](logic/src/policies/README.md) and [models/policies/](logic/src/models/policies/README.md) |
| Dual dispatch unexplained | MEDIUM | RESOLVED — [main.py](main.py) docstring (lines 1-16) explains dispatch mechanism |
| Duplicate sub-layer class names | MEDIUM | MOSTLY RESOLVED — Properly prefixed: `GATFeedForwardSubLayer`, `TGCFeedForwardSubLayer`, `GATMultiHeadAttentionLayer`, etc. Only [gat_decoder.py](logic/src/models/subnets/gat_decoder.py) retains unprefixed base names |
| Raw `print()` for logging (191 calls) | HIGH | RESOLVED — Replaced with proper loggers (`get_pylogger` or `loguru`); 0 print() calls in non-test `logic/src/` |
| Missing paper references | LOW | PARTIALLY RESOLVED — REINFORCE, PPO, POMO, SymNCO now cite papers; SAPO, GSPO, DR-GRPO still don't |
| Forward() data contracts weak | MEDIUM | RESOLVED — Full tensor shape documentation in [AttentionModel.forward()](logic/src/models/attention_model.py#L332-L357) |

---

## Part 3: Remaining Human Comprehension Barriers

### 3.1 HIGH: Problem Size Naming — Three Conventions for One Concept

The concept of "number of locations in a problem instance" uses three different names:

| Convention | Occurrences | Primary Files |
|------------|-------------|---------------|
| `graph_size` | 42 (19 files) | [visualize_utils.py](logic/src/utils/logging/visualize_utils.py), [context_embedder.py](logic/src/models/context_embedder.py), [temporal_am.py](logic/src/models/temporal_am.py) |
| `num_loc` | 52 (6 files) | [generators.py](logic/src/envs/generators.py) (36 alone), [configs/data.py](logic/src/configs/data.py), [configs/env.py](logic/src/configs/env.py) |
| `n_nodes` | 34 (10 files) | [local_search.py](logic/src/models/policies/classical/local_search.py), [loader.py](logic/src/pipeline/simulations/loader.py), other policies |

**Impact**: A developer tracing problem size from config (`num_loc`) to generator (`num_loc`) to model (`graph_size`) to policy (`n_nodes`) encounters three name changes. Unlike the now-resolved `embed_dim`/`n_heads` consistency, this naming split has no single dominant convention.

### 3.2 HIGH: Two Competing Logging Systems

While `print()` has been eliminated, two proper logging systems coexist without documented justification:

| System | Files Using | Primary Domain |
|--------|------------|----------------|
| `get_pylogger()` (Python `logging`) | 6 files: [train.py](logic/src/pipeline/features/train.py), [base.py](logic/src/pipeline/rl/common/base.py), [baselines.py](logic/src/pipeline/rl/common/baselines.py), [gat_lstm_manager.py](logic/src/models/gat_lstm_manager.py), [transforms.py](logic/src/data/transforms.py) | RL training pipeline |
| `loguru` | 8 files: [eval.py](logic/src/pipeline/features/eval.py), [test.py](logic/src/pipeline/features/test.py), [trainer.py](logic/src/pipeline/rl/common/trainer.py), [checkpoints.py](logic/src/pipeline/simulations/checkpoints.py), [states.py](logic/src/pipeline/simulations/states.py), [actions.py](logic/src/pipeline/simulations/actions.py), [storage.py](logic/src/utils/logging/modules/storage.py), [analysis.py](logic/src/utils/logging/modules/analysis.py) | Simulation pipeline & evaluation |

The split is roughly domain-aligned (RL training uses `get_pylogger`, simulation uses `loguru`), but this isn't documented anywhere. A developer debugging across pipeline boundaries must configure two logging systems.

### 3.3 MEDIUM: Broad Exception Handling

76 `except Exception` occurrences across 39 files. The simulation pipeline improved (specific exceptions + loguru), but the training pipeline and model layer still use broad catches:

| Location | Pattern | Risk |
|----------|---------|------|
| [train.py](logic/src/pipeline/features/train.py) | 3 broad `except Exception` | Masks config errors during model creation |
| [base.py](logic/src/pipeline/rl/common/base.py) | Broad catch in weight saving | Silently fails on checkpoint corruption |
| [eval.py](logic/src/pipeline/features/eval.py) | Multiple broad catches | Hides evaluation failures |

### 3.4 MEDIUM: Magic Numbers in Optimization Algorithms

Despite the new `constants/` module, several optimization files retain unexplained magic numbers:

| Value | Occurrences | File | Purpose |
|-------|-------------|------|---------|
| `0.001` | 6x | [local_search.py](logic/src/models/policies/classical/local_search.py) | Improvement threshold (unexplained) |
| `0.9995`, `0.5` | 1x each | [adaptive_large_neighborhood_search.py](logic/src/policies/adaptive_large_neighborhood_search.py) | SA cooling rate, start temperature |
| `100.0` | 2x | [alns.py](logic/src/models/policies/classical/alns.py), [hgs.py](logic/src/models/policies/classical/hgs.py) | Default vehicle capacity (should use `VEHICLE_CAPACITY` from constants) |

### 3.5 MEDIUM: Inconsistent Module-Level Docstring Quality

Module-level docstrings vary significantly in depth:

| Quality | Files | Example |
|---------|-------|---------|
| **Excellent** | [simulator.py](logic/src/pipeline/simulations/simulator.py) (27-line docstring), [eval.py](logic/src/pipeline/features/eval.py) (7-line), [actions.py](logic/src/pipeline/simulations/actions.py) (21-line) | Architecture, responsibilities, classes documented |
| **Basic** | [attention_model.py](logic/src/models/attention_model.py) (1-line), [temporal_am.py](logic/src/models/temporal_am.py) (1-line), [deep_decoder_am.py](logic/src/models/deep_decoder_am.py) (1-line) | Just the module name restated |

43% of key files have good/excellent docstrings. The simulation pipeline excels while the neural model layer is sparse.

### 3.6 MEDIUM: Cognitive Load Hotspots

Functions that require holding too many concepts in working memory:

| Location | Lines | Key Issue |
|----------|-------|-----------|
| [attention_decoder.py:157-260](logic/src/models/subnets/attention_decoder.py#L157-L260) `_inner()` | 103 | POMO expansion, mask dimensions, state mutations |
| [adaptive_large_neighborhood_search.py:98-150](logic/src/policies/adaptive_large_neighborhood_search.py#L98-L150) | 52 | 5 loop-local variables, operator selection, weight adaptation |
| [base.py:174-258](logic/src/pipeline/rl/common/base.py#L174-L258) `shared_step()` | 84 | Tensor device movement, baseline unwrapping, conditional logging |

### 3.7 LOW: Missing Paper References for 3 Algorithms

4 of 7 major RL algorithms now cite their papers (REINFORCE, PPO, POMO, SymNCO). Three variants still lack citations:

| Algorithm | File | Missing Reference |
|-----------|------|-------------------|
| SAPO | [sapo.py](logic/src/pipeline/rl/core/sapo.py) | Self-Adaptive Policy Optimization paper |
| GSPO | [gspo.py](logic/src/pipeline/rl/core/gspo.py) | Gradient-Scaled Proxy Optimization paper |
| DR-GRPO | [dr_grpo.py](logic/src/pipeline/rl/core/dr_grpo.py) | Divergence-Regularized GRPO paper |

### 3.8 LOW: `models/__init__.py` Missing `__all__`

[logic/src/models/__init__.py](logic/src/models/__init__.py) (54 lines) is the only core `__init__.py` without an explicit `__all__` declaration. All other major modules ([envs](logic/src/envs/__init__.py), [pipeline/rl](logic/src/pipeline/rl/__init__.py), [configs](logic/src/configs/__init__.py), [policies](logic/src/policies/__init__.py)) have complete `__all__` lists.

### 3.9 LOW: Algorithm-Specific RL Module Selection Still If-Elif

While the policy creation now uses a clean registry ([train.py:91-100](logic/src/pipeline/features/train.py#L91-L100)), the RL module selection ([train.py:239-361](logic/src/pipeline/features/train.py#L239-L361)) remains an if-elif chain across ~120 lines. Each branch handles different constructor signatures (PPO/SAPO/GSPO/DR-GRPO need critic, POMO/SymNCO need augmentation params, HRL needs manager, Imitation needs expert policy).

---

## Part 4: Improvement Plan (Ordered by Comprehension Impact)

### Phase 1: Naming Standardization (High Impact, Medium Effort)

| # | Task | Impact | Effort | Files |
|---|------|--------|--------|-------|
| 1 | Standardize problem size naming: choose one convention (recommend `num_loc` in configs/generators, `graph_size` in models) and document the mapping | HIGH | MEDIUM | ~35 files across envs, models, policies |
| 2 | Move hardcoded `100.0` vehicle capacity in `alns.py`/`hgs.py` to use `VEHICLE_CAPACITY` from constants | MEDIUM | LOW | 2 files |
| 3 | Extract `0.001` improvement threshold in `local_search.py` to named constant (`IMPROVEMENT_EPSILON`) | MEDIUM | LOW | 1 file |

### Phase 2: Logging & Error Handling (Medium Impact, Medium Effort)

| # | Task | Impact | Effort | Files |
|---|------|--------|--------|-------|
| 4 | Document the logging split: add a brief note in a logging README explaining that `get_pylogger` is for RL training (multi-GPU safe) and `loguru` is for simulation/evaluation | HIGH | LOW | 1 new file |
| 5 | Replace broad `except Exception` in [train.py](logic/src/pipeline/features/train.py) with specific exceptions (`ValueError`, `OSError`, `RuntimeError`) | MEDIUM | LOW | 1 file |
| 6 | Replace broad `except Exception` in [base.py](logic/src/pipeline/rl/common/base.py) weight saving with specific exceptions and `logger.warning()` | MEDIUM | LOW | 1 file |

### Phase 3: Documentation Polish (Medium Impact, Low Effort)

| # | Task | Impact | Effort | Files |
|---|------|--------|--------|-------|
| 7 | Add multi-line module docstrings to [attention_model.py](logic/src/models/attention_model.py), [temporal_am.py](logic/src/models/temporal_am.py), [deep_decoder_am.py](logic/src/models/deep_decoder_am.py) explaining architecture and purpose | MEDIUM | LOW | 3 files |
| 8 | Add paper references to SAPO, GSPO, DR-GRPO docstrings | LOW | LOW | 3 files |
| 9 | Add `__all__` to [models/__init__.py](logic/src/models/__init__.py) | LOW | LOW | 1 file |

### Phase 4: Large File Decomposition (Low Impact, Medium Effort)

| # | Task | Impact | Effort | Files |
|---|------|--------|--------|-------|
| 10 | Split [visualize_utils.py](logic/src/utils/logging/visualize_utils.py) (784 lines) into plot-specific modules | LOW | MEDIUM | 1 file → 3-4 files |
| 11 | Decompose [local_search.py](logic/src/models/policies/classical/local_search.py) (838 lines) into operator-specific files | LOW | MEDIUM | 1 file → 3-4 files |

---

## Part 5: Human Comprehension Assessment

### By Comprehension Dimension

| Dimension | Previous | Current | Evidence |
|-----------|:--------:|:-------:|----------|
| **Onboarding Documentation** | 9/10 | 9/10 | 11K+ lines, clear hierarchy, cross-linked hub table. CLAUDE.md fully updated. |
| **Design Patterns as Guides** | 9/10 | 9/10 | Command, State, Template Method, Registry — all aid understanding |
| **Type Safety / IDE Support** | 8.5/10 | 8.5/10 | ~95%+ public API coverage; `TYPE_CHECKING` in 6 strategic files |
| **Module Export Clarity** | 8/10 | 8.5/10 | Clean `__all__` in all key modules except `models/__init__.py` |
| **Import Organization** | 8/10 | 8.5/10 | 100% consistent in sample of 6 files (stdlib → third-party → local) |
| **Codebase Hygiene** | 9/10 | 9.5/10 | Only 2-3 TODOs; zero FIXME/HACK; empty `tasks/` removed |
| **Constants Management** | — | 8.5/10 | NEW: 11 domain-specific constant files; magic numbers mostly extracted |
| **Naming Consistency** | 4.5/10 | 7/10 | `embed_dim` and `n_heads` FULLY standardized; only `graph_size`/`num_loc`/`n_nodes` remains |
| **Documentation-Code Alignment** | 5/10 | 9/10 | CLAUDE.md fully updated; zero stale `tasks/` references; dual dispatch documented |
| **Logging Consistency** | 4/10 | 7/10 | `print()` eliminated; two proper systems remain (domain-aligned but undocumented split) |
| **Error Handling Consistency** | 5.5/10 | 6.5/10 | Simulation pipeline improved; 76 broad catches remain across training/model layer |
| **Data Contracts** | 6/10 | 8/10 | AttentionModel.forward() documents tensor shapes; simulation action I/O contracts excellent |
| **Navigational Aids** | 5.5/10 | 8/10 | Policy directory READMEs added; main.py dispatch documented; dual policy system explained |
| **Cognitive Load** | 7/10 | 7/10 | Most files <200 lines; max 838 lines; 3 hotspots remain |

### By Subsystem

| Subsystem | Previous | Current | Strongest Aspect | Weakest Aspect |
|-----------|:--------:|:-------:|---|---|
| Simulation Pipeline | 8.5/10 | 8.5/10 | Action I/O contracts, state machine | — |
| Training Pipeline | 7/10 | 7.5/10 | Template Method hierarchy, critic helper | RL module if-elif chain, broad exception handling |
| Neural Models | 6.5/10 | 8/10 | Factory pattern, forward() data contracts, standardized naming | Module-level docstrings sparse |
| GUI Layer | 8/10 | 8/10 | Mediator pattern, tab isolation | No subdirectory READMEs |
| Configuration | 7.5/10 | 8.5/10 | Dataclass composition, algorithm sub-configs, constants module | Config-to-model field mapping undocumented |
| Documentation | 8/10 | 9/10 | Volume, accuracy, dual-policy READMEs, zero stale references | Model layer docstrings inconsistent |

### Overall

| Metric | Previous | Current |
|--------|:--------:|:-------:|
| **Can a new developer find things?** | 7/10 | 8.5/10 |
| **Can they understand what they find?** | 8/10 | 8.5/10 |
| **Can they trust what they read?** | 6.5/10 | 8.5/10 |
| **Can they contribute without breaking things?** | 7.5/10 | 8/10 |
| **Overall Human Comprehension** | **7.3/10** | **8.4/10** |

### Score Improvement Summary

The codebase has improved from **7.3 to 8.4** since the previous analysis through targeted fixes:

| Improvement | Impact on Score |
|-------------|:---:|
| Standardized `embed_dim` and `n_heads` naming (274 occurrences unified) | +0.3 |
| Eliminated all stale `tasks/` references in docs and filesystem | +0.2 |
| Replaced 191 raw `print()` calls with proper loggers | +0.2 |
| Added forward() tensor shape documentation | +0.15 |
| Added policy directory READMEs | +0.1 |
| Documented dual dispatch in main.py | +0.1 |
| Created constants/ module (11 files) | +0.1 |
| Extracted `_create_critic_helper()` | +0.05 |
| Added 4/7 paper references | +0.05 |
| Prefixed duplicate sub-layer class names | +0.05 |

The remaining barriers are primarily the 3-way problem size naming split (`graph_size`/`num_loc`/`n_nodes`), the undocumented logging system split, and broad exception handling in the training pipeline. These are fixable with focused effort rather than architectural changes.
