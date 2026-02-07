# Human Understanding Roadmap

> **Version**: 1.0
> **Date**: 2026-02-07
> **Purpose**: Targeted improvements to make the WSmart-Route codebase easier for humans to read, navigate, and contribute to.
> **Scope**: Documentation, naming, complexity reduction, type safety, onboarding.

---

## Executive Summary

An audit of the WSmart-Route codebase (532 Python files in `logic/src/`, 73 in `gui/src/`) identified the following human-understanding scores:

| Dimension | Score | Key Finding |
|-----------|-------|-------------|
| **Documentation (docstrings)** | 7.5/10 | Logic layer excellent; GUI and env state methods lack coverage |
| **Naming & Consistency** | 8.5/10 | Registries, imports, constants all uniform; minor abbreviation gaps |
| **Code Complexity** | 7/10 | Clean architecture; encoder/decoder boilerplate duplication; deep nesting in test engine |
| **Type Safety** | 8/10 | 95% return-type coverage; magic numbers and asserts need cleanup |
| **Onboarding & Config** | 6.5/10 | Excellent README; config files lack inline docs; dual CLI confusing |

**Overall: 7.5/10** -- a strong foundation with focused improvement opportunities.

---

## Phase H1: Critical Fixes (Immediate)

_Items that actively block or confuse new contributors._

### H1.1 Fix Python Version Constraint

**File**: `pyproject.toml`

`requires-python = ">=3.9, <3.10"` restricts the project to Python 3.9.x only, silently breaking setup for anyone on Python 3.10+.

- [ ] Change to `requires-python = ">=3.9"` (or the actual minimum supported version)
- [ ] Verify `uv sync` works on Python 3.10, 3.11, 3.12

**Severity**: CRITICAL -- blocks onboarding for most users.

---

### H1.2 Document the Dual CLI System

**File**: `main.py` (lines 212-228)

The codebase has two command-routing systems (Hydra-based and legacy argparse) with no guidance on which to use.

- [ ] Add a top-of-file comment block explaining why two systems coexist
- [ ] Document which commands use which system in the docstring
- [ ] Add a `# MIGRATION NOTE` comment explaining the intended future state

---

### H1.3 Add Docstrings to Environment State Methods

**Files**: `logic/src/envs/vrpp.py`, `logic/src/envs/wcvrp.py`

The core state-transition methods (`_reset_instance`, `_step_instance`) have no docstrings, yet they are the most critical logic for understanding the problem physics.

- [ ] `VRPPEnv._reset_instance()` -- document state initialization, TensorDict keys produced
- [ ] `VRPPEnv._step_instance()` -- document action execution, reward logic, state update
- [ ] `WCVRPEnv._reset_instance()` -- document bin initialization, fill levels
- [ ] `WCVRPEnv._step_instance()` -- document capacity checking, collection mechanics
- [ ] `WCVRPEnv._get_action_mask()` -- extend existing docstring with edge-case behavior

---

## Phase H2: Configuration & Onboarding Documentation (Week 1-2)

_Make the first 30 minutes easier for new contributors._

### H2.1 Add Inline Comments to YAML Configs

**Target**: `assets/configs/`

Config parameters currently have no explanation. A newcomer cannot tell what `embed_dim: 128` means or what alternatives exist.

- [ ] `assets/configs/model/am.yaml` -- annotate each parameter with description and valid range
- [ ] `assets/configs/tasks/train.yaml` -- annotate training parameters, explain must-go strategies
- [ ] `assets/configs/envs/*.yaml` -- annotate environment parameters
- [ ] `assets/configs/config.yaml` -- document the Hydra defaults composition

Example of desired state:
```yaml
model:
  name: "am"
  embed_dim: 128          # Node embedding dimension. Powers of 2 recommended (64, 128, 256).
  hidden_dim: 512         # Feed-forward hidden size. Typically 4x embed_dim.
  n_encode_layers: 3      # Transformer encoder depth. More layers = slower but richer encoding.
  normalization: "instance"  # Options: "batch", "layer", "instance", "group". Instance works best for VRP.
```

---

### H2.2 Create Notebook Index

**File**: `notebooks/README.md` (NEW)

16 notebooks exist but there is no guide to which one to read first.

- [ ] Create `notebooks/README.md` with notebooks organized by audience:
  - Beginner: `lightning_rl_training_tutorial.ipynb`
  - Data: `datasets.ipynb`
  - Policies: `VRP_Policy_Regular3day.ipynb`
  - Optimization: `optimization.ipynb`
  - Advanced: `VPP_OneFlow_Lookahead_Dynamic.ipynb`
- [ ] Add difficulty labels and estimated reading time
- [ ] Cross-reference from root `README.md`

---

### H2.3 Document Constants

**Target**: `logic/src/constants/`

11 files of constants (OPERATION_MAP, STATS_FUNCTION_MAP, etc.) with minimal documentation.

- [ ] `system.py` -- add module docstring explaining OPERATION_MAP's purpose and usage context
- [ ] `stats.py` -- add per-function descriptions to STATS_FUNCTION_MAP
- [ ] `policies.py` -- document ENGINE_POLICIES, THRESHOLD_POLICIES with when-to-use guidance
- [ ] `models.py` -- preserve and expand the `num_loc` vs `graph_size` vs `n_nodes` semantic note
- [ ] `simulation.py` -- document simulator parameters with units and valid ranges

---

### H2.4 Create Hydra Config Guide

**File**: `docs/HYDRA_GUIDE.md` (NEW)

No documentation exists for the Hydra config system, override syntax, or composition rules.

- [ ] Explain the defaults composition order (`config.yaml` → `envs/` → `model/` → `tasks/`)
- [ ] Show CLI override syntax: `python main.py train model.embed_dim=256 env.num_loc=100`
- [ ] Document multi-run syntax for sweeps
- [ ] List all available config groups with descriptions

---

## Phase H3: Docstring & Comment Coverage (Week 2-3)

_Close the documentation gaps identified in the audit._

### H3.1 GUI Module Docstrings

**Target**: `gui/src/` package `__init__.py` files

GUI modules are missing module-level documentation (67% coverage vs 95% in logic).

- [ ] `gui/src/__init__.py` -- add description of the GUI layer, list subpackages
- [ ] `gui/src/core/__init__.py` -- describe mediator pattern, signals architecture
- [ ] `gui/src/styles/__init__.py` -- describe theming system (colors, effects, widgets)
- [ ] `gui/src/tabs/__init__.py` -- list all functional tabs with one-line descriptions
- [ ] `gui/src/components/__init__.py` -- list reusable widgets

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

## Phase H4: Code Complexity Reduction (Week 3-4)

_Reduce nesting, duplication, and cognitive load._

### H4.1 Extract Encoder/Decoder Base Classes

**Target**: `logic/src/models/subnets/encoders/`, `logic/src/models/subnets/decoders/`

16 encoders share ~70-80% boilerplate (layer stacking, normalization, dropout). 6+ decoders repeat the same `FeedForwardSubLayer` pattern.

- [ ] Create `logic/src/models/subnets/encoders/base.py` with `TransformerEncoderBase`:
  - Common `__init__` parameter handling (n_heads, embed_dim, n_layers, normalization, activation)
  - Standard layer stacking in `forward()`
  - Subclasses override `_create_layer()` method
- [ ] Create `logic/src/models/subnets/decoders/base.py` with `DecoderFeedForwardSubLayer`
- [ ] Refactor existing encoders to inherit from base (one at a time, test between each)
- [ ] Verify all existing tests pass after each refactoring step

**Impact**: Reduces ~2000 lines of boilerplate across 22 files.

---

### H4.2 Reduce Nesting in Test Engine

**File**: `logic/src/pipeline/features/test/engine.py`

Currently reaches 7 levels of nesting in the multiprocessing orchestration loop.

- [ ] Extract `_parse_policy_config(policy_name, opts)` function for policy configuration parsing (lines 268-369)
- [ ] Extract `_run_parallel_policies(policies, args, opts)` function for multiprocessing management (lines 121-170)
- [ ] Extract `_aggregate_results(results, output_dir)` function for result collection
- [ ] Verify max nesting depth is ≤ 4 after refactoring

---

### H4.3 Split Oversized Utility Files

**File**: `logic/src/policies/look_ahead_aux/select.py` (543 lines, 12 functions)

This monolithic file contains unrelated operations grouped together.

- [ ] Split into:
  - `remove_ops.py` -- bin removal functions
  - `insert_ops.py` -- bin insertion/addition functions
  - `route_ops.py` -- route creation and modification
- [ ] Update imports across codebase
- [ ] Verify policy tests pass

---

## Phase H5: Type Safety & Magic Number Cleanup (Week 4-5)

_Improve static analysis and reduce ambiguity._

### H5.1 Extract Magic Numbers to Named Constants

31 magic numbers found across the codebase. The most impactful to fix:

- [ ] Define `NUMERICAL_EPSILON = 1e-8` in `logic/src/constants/models.py` and use in `decoding/strategies.py`
- [ ] Define `DEFAULT_EVAL_BATCH_SIZE = 1024` in `logic/src/constants/optimization.py`
- [ ] Define `DEFAULT_ROLLOUT_BATCH_SIZE = 64` in `logic/src/constants/optimization.py`
- [ ] Replace hardcoded batch sizes in `pipeline/features/eval/engine.py` and `baselines/rollout.py`
- [ ] Add `# Gurobi: suppress solver output` comment to OutputFlag = 0 in `setup/env.py`

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

- [ ] Choose one style and document in CLAUDE.md coding standards
- [ ] Run automated migration with `pyupgrade --py39-plus` if targeting 3.9
- [ ] Add type hints to GUI helper classes (currently ~70% coverage vs 95% in logic)
- [ ] Consider adding 3 Protocol classes: `DataLike`, `PolicyLike`, `DatasetLike` for duck typing

---

## Phase H6: Naming & Consistency Polish (Ongoing)

_Low-priority improvements that compound over time._

### H6.1 File Naming Glossary

**Target**: `logic/src/models/modules/`, `logic/src/models/subnets/`

Some file names use abbreviations (`ptr`, `moe`, `hgnn`, `mpnn`) that are not immediately clear.

- [ ] Add an abbreviation glossary to `CLAUDE.md` Section 6 (or a standalone `GLOSSARY.md`)
- [ ] Include: `ptr` = Pointer, `moe` = Mixture of Experts, `hgnn` = Heterogeneous GNN, `mpnn` = Message Passing NN, `mdam` = Multi-Decoder Attention Model, `nar` = Non-Autoregressive, `l2d` = Learning to Dispatch

---

### H6.2 Variable Naming Standards

Documented best practices to reduce future ambiguity:

- [ ] Document that single-letter tensor variables (`B`, `N`, `u`, `v`) are acceptable in algorithmic code but must have a shape comment: `B, N = parent1.size()  # B=batch, N=nodes`
- [ ] Standardize weight matrix naming: `W_query` (capital W for learnable parameters)
- [ ] Standardize tensor flattening variables: `h_flat` not `hflat`
- [ ] Add to CLAUDE.md Section 6.3

---

### H6.3 Reduce Parameter Sprawl in Constructors

Some constructors have 20+ parameters (e.g., `AttentionModel.__init__`, `GATDecoder.__init__`).

- [ ] Group related parameters into dataclasses:
  - `NormalizationConfig(type, epsilon, learn_affine, track_stats, momentum, n_groups)`
  - `ActivationConfig(name, param, threshold, replacement_value, n_params, range)`
- [ ] Apply to encoders and decoders with 10+ normalization/activation parameters
- [ ] Maintain backward compatibility via `**kwargs` expansion

---

## Phase H7: Problem-Model Compatibility Matrix (Docs)

_Help users pick the right model for their problem._

### H7.1 Create COMPATIBILITY.md

**File**: `COMPATIBILITY.md` (NEW)

- [ ] Model-Problem support matrix (which models work with which env names)
- [ ] Encoder-Decoder compatibility table
- [ ] RL Algorithm-Policy type compatibility (constructive, improvement, transductive, classical)
- [ ] Recommended configurations per problem type with expected training times
- [ ] Link from README.md and CLAUDE.md

---

## Timeline Summary

| Phase | Focus | Priority | Effort |
|-------|-------|----------|--------|
| H1 | Critical fixes (Python version, CLI docs, env docstrings) | CRITICAL | 1-2 days |
| H2 | Config & onboarding docs (YAML comments, notebook index, Hydra guide) | HIGH | 3-5 days |
| H3 | Docstring gap closure (GUI modules, functions, inline comments) | HIGH | 3-5 days |
| H4 | Complexity reduction (base classes, nesting, file splits) | MEDIUM | 5-7 days |
| H5 | Type safety & magic numbers (constants, asserts, type hints) | MEDIUM | 3-5 days |
| H6 | Naming polish (glossary, variable standards, parameter grouping) | LOW | 2-3 days |
| H7 | Compatibility matrix documentation | LOW | 1-2 days |

---

## Metrics & Targets

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Module docstring coverage | 85% | 95% | `just check-docs` |
| Function docstring coverage | 78% | 90% | `just check-docs` |
| Type hint coverage (returns) | 95% | 98% | `mypy --strict` |
| Type hint coverage (params) | 90% | 95% | `mypy --strict` |
| Files > 500 LOC | 9 | ≤ 5 | `find -name "*.py" \| xargs wc -l \| awk '$1>500'` |
| Max nesting depth | 7 | ≤ 4 | Manual audit / radon |
| Magic numbers (undocumented) | 31 | ≤ 10 | grep audit |
| Asserts without messages | 25 | 0 | `grep -r "assert " \| grep -v "#"` |
| Config params without comments | ~80% | ≤ 20% | Manual audit of `assets/configs/` |

---

## Cross-References

- [ROADMAP.md](ROADMAP.md) -- Feature implementation roadmap (rl4co parity, Phases 1-14)
- [CLAUDE.md](CLAUDE.md) -- Coding standards and AI assistant instructions
- [ARCHITECTURE.md](ARCHITECTURE.md) -- System design documentation
- [DEVELOPMENT.md](DEVELOPMENT.md) -- Developer environment setup
- [TESTING.md](TESTING.md) -- Test suite organization
