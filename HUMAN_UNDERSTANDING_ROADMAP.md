# Human Understanding Roadmap

> **Version**: 1.0
> **Date**: 2026-02-07
> **Purpose**: Targeted improvements to make the WSmart-Route codebase easier for humans to read, navigate, and contribute to.
> **Scope**: Documentation, naming, complexity reduction, type safety, onboarding.

---

## Executive Summary

An audit of the WSmart-Route codebase (532 Python files in `logic/src/`, 73 in `gui/src/`) identified the following human-understanding scores:

| Dimension                      | Score  | Key Finding                                                                              |
| ------------------------------ | ------ | ---------------------------------------------------------------------------------------- |
| **Documentation (docstrings)** | 7.5/10 | Logic layer excellent; GUI and env state methods lack coverage                           |
| **Naming & Consistency**       | 8.5/10 | Registries, imports, constants all uniform; minor abbreviation gaps                      |
| **Code Complexity**            | 7/10   | Clean architecture; encoder/decoder boilerplate duplication; deep nesting in test engine |
| **Type Safety**                | 8/10   | 95% return-type coverage; magic numbers and asserts need cleanup                         |
| **Onboarding & Config**        | 6.5/10 | Excellent README; config files lack inline docs; dual CLI confusing                      |

**Overall: 7.5/10** -- a strong foundation with focused improvement opportunities.

---

## Implementation Progress (Updated 2026-02-08)

| Phase     | Focus                        | Status           | Tasks Complete |
| --------- | ---------------------------- | ---------------- | -------------- |
| H1        | Critical Fixes               | ✅ COMPLETE      | 3/3 (100%)     |
| H2        | Configuration & Onboarding   | ✅ COMPLETE      | 4/4 (100%)     |
| H3        | Docstring & Comment Coverage | ⚠️ Pending       | 0/3 (0%)       |
| H4        | Complexity Reduction         | 🔄 IN PROGRESS   | 1/2 (50%)      |
| H5        | Type Safety & Magic Numbers  | 🔄 IN PROGRESS   | 1/3 (33%)      |
| H6        | Naming & Consistency Polish  | ⚠️ Pending       | 0/3 (0%)       |
| H7        | Compatibility Matrix         | ⚠️ Pending       | 0/1 (0%)       |
| **TOTAL** | **All Phases**               | **47% Complete** | **9/19 tasks** |

**Recently Completed**: Phase H4.1 (Base classes) + H5.1 (Magic numbers)
**Next Recommended**: Phase H4.2 (Reduce nesting) OR Phase H5.2 (Replace asserts) - 30-60 minutes each

See [ROADMAP_PROGRESS.md](ROADMAP_PROGRESS.md) for detailed session notes and file modifications.

---

## ✅ Phase H1: Critical Fixes (COMPLETE)

_Items that actively block or confuse new contributors._

### H1.1 Fix Python Version Constraint ✅

**File**: `pyproject.toml`

`requires-python = ">=3.9, <3.10"` restricts the project to Python 3.9.x only, silently breaking setup for anyone on Python 3.10+.

- [x] Change to `requires-python = ">=3.9"` (or the actual minimum supported version)
- [x] Verify `uv sync` works on Python 3.10, 3.11, 3.12

**Severity**: CRITICAL -- blocks onboarding for most users.
**Status**: ✅ COMPLETE

---

### H1.2 Document the Dual CLI System ✅

**File**: `main.py` (lines 212-228)

The codebase has two command-routing systems (Hydra-based and legacy argparse) with no guidance on which to use.

- [x] Add a top-of-file comment block explaining why two systems coexist
- [x] Document which commands use which system in the docstring
- [x] Add a `# MIGRATION NOTE` comment explaining the intended future state

**Status**: ✅ COMPLETE - Added comprehensive docstring to `main_dispatch()` with MIGRATION NOTE

---

### H1.3 Add Docstrings to Environment State Methods ✅

**Files**: `logic/src/envs/vrpp.py`, `logic/src/envs/wcvrp.py`

The core state-transition methods (`_reset_instance`, `_step_instance`) have no docstrings, yet they are the most critical logic for understanding the problem physics.

- [x] `VRPPEnv._reset_instance()` -- document state initialization, TensorDict keys produced
- [x] `VRPPEnv._step_instance()` -- document action execution, reward logic, state update
- [x] `WCVRPEnv._reset_instance()` -- document bin initialization, fill levels
- [x] `WCVRPEnv._step_instance()` -- document capacity checking, collection mechanics
- [x] `WCVRPEnv._get_action_mask()` -- extend existing docstring with edge-case behavior

**Status**: ✅ COMPLETE - All 5 methods now have comprehensive docstrings (35-45 lines each)

---

## ✅ Phase H2: Configuration & Onboarding Documentation (COMPLETE)

_Make the first 30 minutes easier for new contributors._

### H2.1 Add Inline Comments to YAML Configs ✅

**Target**: `assets/configs/`

Config parameters currently have no explanation. A newcomer cannot tell what `embed_dim: 128` means or what alternatives exist.

- [x] `assets/configs/config.yaml` -- document the Hydra defaults composition
- [x] `assets/configs/model/am.yaml` -- annotate each parameter with description and valid range (90+ lines)
- [x] `assets/configs/tasks/train.yaml` -- annotate training parameters, explain must-go strategies
- [x] `assets/configs/envs/*.yaml` -- annotate environment parameters (cwcvrp, wcvrp, vrpp, etc.)
- [x] Other model configs (tam.yaml, deep_decoder.yaml, ptr.yaml, etc.)

**Status**: ✅ COMPLETE - All 15+ config files now have inline documentation

Example of completed state (am.yaml):

```yaml
model:
  name: "am"
  embed_dim: 128 # Node embedding dimension. Powers of 2 recommended (64, 128, 256).
  hidden_dim: 512 # Feed-forward hidden size. Typically 4x embed_dim.
  n_encode_layers: 3 # Transformer encoder depth. More layers = slower but richer encoding.
  normalization: "instance" # Options: "batch", "layer", "instance", "group". Instance works best for VRP.
```

---

### H2.2 Create Notebook Index ✅

**File**: `notebooks/README.md` (NEW - 230 lines)

16 notebooks exist but there is no guide to which one to read first.

- [x] Create `notebooks/README.md` with notebooks organized by audience:
  - Beginner: `lightning_rl_training_tutorial.ipynb`
  - Data: `datasets.ipynb`
  - Policies: `VRP_Policy_Regular3day.ipynb`
  - Optimization: `optimization.ipynb`
  - Advanced: `VPP_OneFlow_Lookahead_Dynamic.ipynb`
- [x] Add difficulty labels and estimated reading time
- [x] Cross-reference from root `README.md`

**Status**: ✅ COMPLETE - Comprehensive notebook index with 7 tutorial sequences, prerequisites, and tips

---

### H2.3 Document Constants ✅

**Target**: `logic/src/constants/`

13 files of constants (OPERATION_MAP, STATS_FUNCTION_MAP, etc.) with minimal documentation.

- [x] `system.py` -- add module docstring explaining OPERATION_MAP's purpose and usage context
- [x] `stats.py` -- add per-function descriptions to STATS_FUNCTION_MAP
- [x] `policies.py` -- document ENGINE_POLICIES, THRESHOLD_POLICIES with when-to-use guidance
- [x] `models.py` -- preserve and expand the `num_loc` vs `graph_size` vs `n_nodes` semantic note
- [x] `simulation.py` -- document simulator parameters with units and valid ranges
- [x] `dashboard.py` -- ColorBrewer2 palette, bin status colors, colorblind-safe design rationale
- [x] `hpo.py` -- HPO configuration keys for Optuna/DEHB/Ray Tune (27 parameters)
- [x] `optimization.py` -- Operational parameters, Gurobi MIP tuning, LAC scheduling defaults
- [x] `paths.py` -- Dynamic root directory resolution, platform-independent path handling
- [x] `tasks.py` -- Legacy constants with deprecation warnings, migration path to config files
- [x] `testing.py` -- Test module registry (14 modules), CLI/component/integration test organization
- [x] `user_interface.py` -- TQDM colors, matplotlib styling, GUI themes, accessibility rationale
- [x] `waste.py` -- Portugal case study metadata, depot mappings, waste type translations

**Status**: ✅ COMPLETE - ALL 13 constants modules fully documented (425 → 1,133 lines, +166% growth)

---

### H2.4 Create Hydra Config Guide ✅

**File**: `docs/HYDRA_GUIDE.md` (NEW - 350+ lines)

No documentation exists for the Hydra config system, override syntax, or composition rules.

- [x] Explain the defaults composition order (`config.yaml` → `envs/` → `model/` → `tasks/`)
- [x] Show CLI override syntax: `python main.py train model.embed_dim=256 env.num_loc=100`
- [x] Document multi-run syntax for sweeps
- [x] List all available config groups with descriptions

**Status**: ✅ COMPLETE - Comprehensive 350+ line guide with 20+ examples, troubleshooting, and use cases

---

## Phase H3: Docstring & Comment Coverage (IN PROGRESS)

_Close the documentation gaps identified in the audit._

### H3.1 GUI Module Docstrings (COMPLETE)

**Target**: `gui/src/` package `__init__.py` files

GUI modules are missing module-level documentation (67% coverage vs 95% in logic).

- [x] `gui/src/__init__.py` -- add description of the GUI layer, list subpackages
- [x] `gui/src/core/__init__.py` -- describe mediator pattern, signals architecture
- [x] `gui/src/styles/__init__.py` -- describe theming system (colors, effects, widgets)
- [x] `gui/src/tabs/__init__.py` -- list all functional tabs with one-line descriptions
- [x] `gui/src/components/__init__.py` -- list reusable widgets

---

### H3.2 Function Docstring Gap Closure

**Target**: Public functions currently at ~75-80% coverage; raise to 90%+.

Priority files (complex logic, high import count):

- [ ] `logic/src/pipeline/features/train/model_factory.py:create_model()` -- document the 3-phase creation (env → policy → RL module)
- [ ] `logic/src/pipeline/features/test/engine.py` -- document `run_policy_test()` and `run_wsr_simulator_test()`
- [ ] `logic/src/policies/neural_agent.py` -- document `compute_batch_sim()`, `compute_simulator_day()`
- [ ] `logic/src/models/model_factory.py` -- document factory extensibility (how to add a new encoder)
- [ ] `gui/src/helpers/chart_worker.py` -- add type hints to constructor and `process_data()`

---

### H3.3 Inline Comments for Complex Algorithms

**Target**: Files with complex logic that currently lack step-by-step commentary.

- [ ] `logic/src/utils/functions/decoding/beam_search.py` -- explain beam queue management, pruning strategy
- [ ] `logic/src/models/policies/classical/hybrid_genetic_search.py` -- explain crossover operator, population diversity metric
- [ ] `logic/src/policies/look_ahead_aux/search.py` -- explain heuristic search strategy with high-level overview
- [ ] `logic/src/pipeline/simulations/wsmart_bin_analysis/Deliverables/simulation.py` -- explain GridBase preprocessing pipeline

---

## ⚠️ Phase H4: Code Complexity Reduction (IN PROGRESS - 1/2)

_Reduce nesting, duplication, and cognitive load._

### H4.1 Extract Encoder/Decoder Base Classes (ONGOING)

**Target**: `logic/src/models/subnets/encoders/`, `logic/src/models/subnets/decoders/`

16 encoders share ~70-80% boilerplate (layer stacking, normalization, dropout). 6+ decoders repeat the same `FeedForwardSubLayer` pattern.

- [x] Create `logic/src/models/subnets/encoders/common/` with `TransformerEncoderBase`:
  - Common `__init__` parameter handling (n_heads, embed_dim, n_layers, normalization, activation)
  - Standard layer stacking in `forward()`
  - Subclasses override `_create_layer()` method
  - File: `encoder_base.py` (226 lines, fully documented with usage examples)
- [x] Create `logic/src/models/subnets/decoders/common/` with `FeedForwardSubLayer`:
  - Reusable feed-forward sublayer for all decoder architectures
  - File: `feed_forward_sublayer.py` (138 lines, fully documented)
- [ ] Refactor existing encoders to inherit from base (DEFERRED - base classes ready for adoption)
- [ ] Verify all existing tests pass after each refactoring step (DEFERRED)

**Status**: ✅ COMPLETE - Base classes created and available for use
**Impact**: Infrastructure ready to reduce ~2000 lines of boilerplate across 22 files when adopted

---

### H4.2 Reduce Nesting in Test Engine

**File**: `logic/src/pipeline/features/test/engine.py`

Currently reaches 7 levels of nesting in the multiprocessing orchestration loop.

- [ ] Extract `_parse_policy_config(policy_name, opts)` function for policy configuration parsing (lines 268-369)
- [ ] Extract `_run_parallel_policies(policies, args, opts)` function for multiprocessing management (lines 121-170)
- [ ] Extract `_aggregate_results(results, output_dir)` function for result collection
- [ ] Verify max nesting depth is ≤ 4 after refactoring

---

## ⚠️ Phase H5: Type Safety & Magic Number Cleanup (IN PROGRESS - 1/3)

_Improve static analysis and reduce ambiguity._

### H5.1 Extract Magic Numbers to Named Constants ✅

31 magic numbers found across the codebase. The most impactful to fix:

- [x] Define `NUMERICAL_EPSILON = 1e-8` in `logic/src/constants/models.py` and use in decoding files
- [x] Define `DEFAULT_EVAL_BATCH_SIZE = 1024` in `logic/src/constants/routing.py`
- [x] Define `DEFAULT_ROLLOUT_BATCH_SIZE = 64` in `logic/src/constants/routing.py`
- [x] Replace hardcoded batch sizes in `models/hrl_manager/manager.py` and `baselines/rollout.py`
- [x] Replace 1e-8 in `utils/decoding/greedy.py`, `utils/decoding/sampling.py`
- [x] Replace 1e-8 in `models/subnets/decoders/mdam/cache.py`, `models/subnets/decoders/polynet/decoder.py`
- [x] Add `# Gurobi: suppress solver output` comment to OutputFlag = 0 in `setup/env.py`

**Status**: ✅ COMPLETE - 8 files modified, 8 magic numbers eliminated, all linter checks passed

---

### H5.2 Replace Validation Asserts with Exceptions

49 assert statements found; 53% lack messages and some are used for user-input validation.

- [ ] `logic/src/utils/configs/setup/env.py:86` -- replace `assert env_filename is not None` with `ValueError`
- [ ] `logic/src/utils/functions/model.py:57` -- replace `assert load_path is None or resume is None` with `ValueError`
- [ ] `logic/src/utils/configs/setup/optimization.py:49,90` -- keep asserts but add descriptive messages
- [ ] Audit remaining 45 asserts: add messages where missing, convert user-facing ones to exceptions

---

### H5.3 Standardize Type Hint Style

The codebase mixes `Tuple[...]` (Python 3.8 style) with `tuple[...]` (Python 3.10+ style).

- [ ] Choose one style and document in AGENTS.md coding standards (use 3.8 style)
- [ ] Run automated migration with `pyupgrade --py39-plus` if targeting 3.9
- [ ] Add type hints to GUI helper classes (currently ~70% coverage vs 95% in logic)
- [ ] Consider adding 3 Protocol classes: `DataLike`, `PolicyLike`, `DatasetLike` for duck typing

---

## Phase H6: Naming & Consistency Polish (Ongoing)

_Low-priority improvements that compound over time._

### H6.1 File Naming Glossary

**Target**: `logic/src/models/modules/`, `logic/src/models/subnets/`

Some file names use abbreviations (`ptr`, `moe`, `hgnn`, `mpnn`) that are not immediately clear.

- [ ] Add an abbreviation glossary to `AGENTS.md` Section 6 (or a standalone `GLOSSARY.md`)
- [ ] Include: `ptr` = Pointer, `moe` = Mixture of Experts, `hgnn` = Heterogeneous GNN, `mpnn` = Message Passing NN, `mdam` = Multi-Decoder Attention Model, `nar` = Non-Autoregressive

---

### H6.2 Variable Naming Standards

Documented best practices to reduce future ambiguity:

- [ ] Document that single-letter tensor variables (`B`, `N`, `u`, `v`) are acceptable in algorithmic code but must have a shape comment: `B, N = parent1.size()  # B=batch, N=nodes`
- [ ] Standardize weight matrix naming: `W_query` (capital W for learnable parameters)
- [ ] Standardize tensor flattening variables: `h_flat` not `hflat`
- [ ] Add to AGENTS.md Section 6.3

---

### H6.3 Reduce Parameter Sprawl in Constructors (ONGOING)

Some constructors have 20+ parameters (e.g., `AttentionModel.__init__`, `GATDecoder.__init__`).

- [x] Group related parameters into dataclasses:
  - `NormalizationConfig(type, epsilon, learn_affine, track_stats, momentum, n_groups)`
  - `ActivationConfig(name, param, threshold, replacement_value, n_params, range)`
- [ ] Apply to encoders and decoders with 10+ normalization/activation parameters

---

## Phase H7: Problem-Model Compatibility Matrix (COMPLETE)

_Help users pick the right model for their problem._

### H7.1 Create COMPATIBILITY.md

**File**: `COMPATIBILITY.md` (NEW)

- [x] Model-Problem support matrix (which models work with which env names)
- [x] Encoder-Decoder compatibility table
- [x] RL Algorithm-Policy type compatibility (constructive, improvement, transductive, classical)
- [x] Recommended configurations per problem type with expected training times
- [x] Link from README.md and AGENTS.md

---

## Timeline Summary

| Phase | Focus                                                                 | Priority | Effort   |
| ----- | --------------------------------------------------------------------- | -------- | -------- |
| H1    | Critical fixes (Python version, CLI docs, env docstrings)             | CRITICAL | 1-2 days |
| H2    | Config & onboarding docs (YAML comments, notebook index, Hydra guide) | HIGH     | 3-5 days |
| H3    | Docstring gap closure (GUI modules, functions, inline comments)       | HIGH     | 3-5 days |
| H4    | Complexity reduction (base classes, nesting, file splits)             | MEDIUM   | 5-7 days |
| H5    | Type safety & magic numbers (constants, asserts, type hints)          | MEDIUM   | 3-5 days |
| H6    | Naming polish (glossary, variable standards, parameter grouping)      | LOW      | 2-3 days |
| H7    | Compatibility matrix documentation                                    | LOW      | 1-2 days |

---

## Metrics & Targets

| Metric                         | Current | Target | How to Measure                                     |
| ------------------------------ | ------- | ------ | -------------------------------------------------- |
| Module docstring coverage      | 85%     | 95%    | `just check-docs`                                  |
| Function docstring coverage    | 78%     | 90%    | `just check-docs`                                  |
| Type hint coverage (returns)   | 95%     | 98%    | `mypy --strict`                                    |
| Type hint coverage (params)    | 90%     | 95%    | `mypy --strict`                                    |
| Files > 500 LOC                | 9       | ≤ 5    | `find -name "*.py" \| xargs wc -l \| awk '$1>500'` |
| Max nesting depth              | 7       | ≤ 4    | Manual audit / radon                               |
| Magic numbers (undocumented)   | 31      | ≤ 10   | grep audit                                         |
| Asserts without messages       | 25      | 0      | `grep -r "assert " \| grep -v "#"`                 |
| Config params without comments | ~80%    | ≤ 20%  | Manual audit of `assets/configs/`                  |

---

## Cross-References

- [ROADMAP.md](ROADMAP.md) -- Feature implementation roadmap (rl4co parity, Phases 1-14)
- [AGENTS.md](AGENTS.md) -- Coding standards and AI assistant instructions
- [ARCHITECTURE.md](ARCHITECTURE.md) -- System design documentation
- [DEVELOPMENT.md](DEVELOPMENT.md) -- Developer environment setup
- [TESTING.md](TESTING.md) -- Test suite organization
