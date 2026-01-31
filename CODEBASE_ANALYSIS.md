# WSmart-Route Codebase Analysis & Improvement Plan

> **Scope**: 345 Python files, ~53,600 lines (logic) + GUI layer
> **Date**: January 2026

---

## Part 1: Strengths

### 1.1 Exemplary Design Pattern Usage

The codebase demonstrates professional-grade pattern adoption that directly aids human comprehension:

- **Command Pattern** ([actions.py](logic/src/pipeline/simulations/actions.py)): Six discrete action classes (`FillAction`, `MustGoSelectionAction`, `PolicyExecutionAction`, `PostProcessAction`, `CollectAction`, `LogAction`) each encapsulate a single simulation step. The pipeline in [day.py:162-172](logic/src/pipeline/simulations/day.py#L162-L172) reads as a literal description of the process: Fill -> Select -> Route -> Refine -> Collect -> Log.

- **State Pattern** ([states.py](logic/src/pipeline/simulations/states.py)): The `InitializingState -> RunningState -> FinishingState` lifecycle is self-documenting. The `SimulationContext.run()` method at [line 277-286](logic/src/pipeline/simulations/states.py#L277-L286) is a textbook state machine loop.

- **Registry + Factory** ([adapters.py](logic/src/policies/adapters.py)): The `@PolicyRegistry.register("name")` decorator pattern makes it trivially easy to discover available policies. `PolicyFactory.get_adapter()` provides a single entry point with well-organized lazy imports.

- **Mediator Pattern** ([mediator.py](gui/src/core/mediator.py)): Clean GUI communication hub preventing tight coupling between tabs and windows.

### 1.2 Strong Documentation Discipline

- **Module-level docstrings**: Nearly all files have comprehensive module docstrings explaining purpose, architecture, and class listings (e.g., [actions.py:1-21](logic/src/pipeline/simulations/actions.py#L1-L21), [states.py:1-30](logic/src/pipeline/simulations/states.py#L1-L30)).
- **Context Input/Output contracts**: Each action class documents exactly what it reads from and writes to the shared context (e.g., `FillAction` at [line 98-117](logic/src/pipeline/simulations/actions.py#L98-L117)). This is exceptional for a dynamically-typed context dictionary.
- **Type hints**: ~90%+ coverage on public functions with proper use of `Optional`, `Tuple`, `Dict`, and `TYPE_CHECKING` imports.

### 1.3 Clean Architectural Boundaries

- **Logic/GUI separation**: The GUI layer imports from Logic but never the reverse. This is verified by the import structure.
- **`SimulationDayContext` dataclass** ([context.py](logic/src/pipeline/simulations/context.py)): Replaces a loosely-typed dictionary with a structured dataclass while maintaining backward compatibility via `Mapping` interface. This is a pragmatic bridge between old and new patterns.
- **Constants module** ([logic/src/constants/](logic/src/constants/)): Domain constants are centralized in separate files by concern (`models.py`, `simulation.py`, `waste.py`, etc.), avoiding magic numbers scattered through business logic.

### 1.4 Mathematical Code Quality

- **Local search operators**: Well-documented vectorized algorithms with complexity notes and tensor shape comments.
- **Attention mechanisms**: Standard ML notation (`Q, K, V`, `B, N`) used consistently where domain conventions apply.
- **Algorithm references**: Code references original papers and methods (e.g., Welford's algorithm in bins, Xavier initialization rationale).

### 1.5 Comprehensive CLAUDE.md/AGENTS.md

The 700+ line [CLAUDE.md](CLAUDE.md) is an unusually thorough developer reference covering architecture, CLI commands, coding standards, severity protocols, and known constraints. This significantly aids both human and AI contributors.

---

## Part 2: Weaknesses

### 2.1 CRITICAL: Policy Layer Code Duplication

**Impact on human understanding**: A developer reading policy adapters sees the same boilerplate 10 times, making it hard to identify what's actually unique about each policy.

All 10+ policy adapters in `logic/src/policies/policy_*.py` repeat identical patterns:

```python
# Repeated in every policy file:
must_go = kwargs.get("must_go", [])
if not must_go:
    return [0, 0], 0.0, None
bins = kwargs["bins"]
distance_matrix = kwargs["distance_matrix"]
# ... load_area_and_waste_type_params() ...
# ... subset matrix with np.ix_() ...
# ... map back to global IDs ...
return tour, get_route_cost(distance_matrix, tour), None
```

**Files affected**: `policy_alns.py`, `policy_bcp.py`, `policy_hgs.py`, `policy_hgs_alns.py`, `policy_tsp.py`, `policy_cvrp.py`, `policy_vrpp.py`, `policy_lkh.py`, `policy_sans.py`, `policy_lac.py`

**Improvement**: Extract a `BaseRoutingPolicy` class with:
- `_validate_must_go()` - common empty-check short-circuit
- `_create_subset_matrix()` - common distance matrix subsetting
- `_map_tour_to_global()` - common index mapping
- `_load_area_params()` - common parameter loading

### 2.2 HIGH: Dual Import Path Structure

**Impact on human understanding**: Developers encounter two different paths for the same concept, causing confusion about which is canonical.

There exist two parallel policy hierarchies:
- `logic/src/policies/` - Simulation-facing policy adapters (used by simulator)
- `logic/src/models/policies/` - Lightning-facing policy wrappers (used by RL training)

```python
# In train.py:
from logic.src.models.policies import AttentionModelPolicy
from logic.src.models.policies.classical.alns import ALNSPolicy

# In adapters.py:
from logic.src.policies.policy_alns import ALNSPolicy
```

Both `ALNSPolicy` classes exist but serve different purposes. This naming collision is confusing.

**Improvement**: Rename one set to clearly distinguish them:
- `logic/src/policies/` -> keep as `PolicyAdapter` classes (simulation interface)
- `logic/src/models/policies/` -> rename classes to `*PolicyModule` or `*PolicyWrapper` (RL interface)

### 2.3 HIGH: SimulationContext Policy Parsing (states.py:142-237)

**Impact on human understanding**: The 95-line chain of `elif "vrpp" in self.pol_strip` / `elif "sans" in self.pol_strip` / `elif "lac" in self.pol_strip` / etc. is a maintenance hazard and hard to read.

The repeated pattern:
```python
try:
    parts = self.pol_strip.split("NAME")
    if len(parts) > 1:
        threshold_part = parts[1].strip("_")
        sub_parts = threshold_part.split("_")
        if sub_parts[0]:
            self.pol_threshold = float(sub_parts[0])
except (ValueError, IndexError):
    pass
```

...appears 5 times verbatim for different policy names.

**Improvement**: Extract a `_parse_policy_string(name, keywords)` method that handles all parsing variants via a lookup table:

```python
POLICY_PARSERS = {
    "vrpp": {"engines": ["gurobi", "hexaly"]},
    "sans": {},
    "lac": {"config_chars": ["a", "b"]},
    "hgs": {},
    "alns": {},
}
```

### 2.4 HIGH: Broad Exception Handling

**Impact on human understanding**: Silent `except Exception:` blocks hide bugs and make debugging frustrating.

Found 15+ instances of overly broad exception catching:

| File | Pattern | Risk |
|------|---------|------|
| `states.py:367` | `except Exception: pass` | Silently ignores config load failures |
| `actions.py:468` | `except Exception as e: print(...)` | Swallows post-processing errors |
| `actions.py:441` | `except Exception as e: print(...); continue` | Hides config file issues |
| Various `policy_*.py` | `except Exception:` | Silent routing failures |
| `crypto_utils.py` (3x) | `except Exception:` | Security-critical silent failures |

**Improvement**: Replace with specific exceptions (`FileNotFoundError`, `ValueError`, `KeyError`) and always log with `logger.warning()` or `logger.error()` instead of `print()`.

### 2.5 MEDIUM: Overuse of `Any` Type in Context Objects

**Impact on human understanding**: `SimulationDayContext` has 20+ fields typed as `Any`, defeating the purpose of the dataclass structure.

```python
# context.py - 20 fields typed as Any:
bins: Any = None
new_data: Any = None
coords: Any = None
distance_matrix: Any = None
model_env: Any = None
device: Any = None
lock: Any = None
hrl_manager: Any = None
# ... etc
```

When everything is `Any`, the dataclass provides structure but not type safety. A reader cannot determine what type `bins` actually is without tracing through the codebase.

**Improvement**: Replace `Any` with actual types:
```python
bins: Optional["Bins"] = None
coords: Optional[pd.DataFrame] = None
distance_matrix: Optional[np.ndarray] = None
device: Optional[torch.device] = None
lock: Optional[mp.Lock] = None
```

### 2.6 MEDIUM: Policy-to-Simulator Upward Dependency

**Impact on human understanding**: Policies conceptually should be independent of the simulation infrastructure, but they import from it.

```python
# In policy_bcp.py, policy_alns.py, policy_hgs.py, etc:
from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params
```

This creates a circular conceptual dependency: policies depend on the simulator that invokes them.

**Improvement**: Move `load_area_and_waste_type_params()` to `logic/src/utils/` or pass the parameters through the context dictionary (they're already available in `SimulationDayContext`).

### 2.7 MEDIUM: Large Files Needing Decomposition

| File | Lines | Issue |
|------|-------|-------|
| `solutions.py` | 1,518 | Multiple responsibilities: route search, tabu management, initialization |
| `container.py` | 905 | Monolithic data container |
| `log_utils.py` | 904 | Logging + visualization mixed |
| `local_search.py` | 838 | 8+ distinct operators in one file |
| `visualize_utils.py` | 784 | Many independent plot functions |
| `states.py` | 777 | InitializingState alone is ~200 lines |
| `swap.py` | 761 | Multiple swap variants |

**Improvement for top 3**:
- `solutions.py` -> split into `route_search.py`, `tabu_management.py`, `solution_initialization.py`
- `log_utils.py` -> split into `log_utils.py` (logging) and `log_visualization.py` (plotting)
- `local_search.py` -> split into `two_opt.py`, `or_opt.py`, `relocate.py`

### 2.8 MEDIUM: Configuration Loading Inconsistency

Three different config loading patterns coexist:

1. **Hydra/OmegaConf** (training pipeline): `@hydra.main()` + `ConfigStore`
2. **YAML file loading** (simulator): `load_config()` from `config_loader.py`
3. **XML file loading** (legacy policies): `load_config()` with XML support

Config files live in multiple locations:
- `scripts/configs/policies/` (policy configs)
- `logic/src/configs/` (structured Python configs)
- `logic/src/utils/configs/` (config loading utilities)

**Improvement**: Standardize on Hydra-style configs where possible. At minimum, document the loading hierarchy and which pattern to use for each use case.

### 2.9 LOW: Backward Compatibility Shims

[adapters.py:149-200](logic/src/policies/adapters.py#L149-L200) contains a 50-line `__getattr__` function providing backward-compatible class aliases. While functional, this creates confusion about canonical names.

**Improvement**: Document a deprecation timeline and migrate callers to canonical names.

### 2.10 LOW: Inconsistent Config Flattening

`_flatten_config()` in [actions.py:33-68](logic/src/pipeline/simulations/actions.py#L33-L68) is called 3 times within the same module for different sections (must_go, policy, post_processing), each time with slightly different expectations about the nested structure.

**Improvement**: Either standardize config structure so flattening isn't needed, or create config-section-specific extractors with clear contracts.

---

## Part 3: Improvement Plan (Prioritized)

### Phase 1: Reduce Confusion (Human Understanding Focus)

| # | Task | Impact | Files |
|---|------|--------|-------|
| 1 | Extract `BaseRoutingPolicy` to eliminate policy adapter duplication | High | 10 `policy_*.py` files |
| 2 | Add proper types to `SimulationDayContext` (replace `Any`) | High | `context.py` |
| 3 | Extract policy parsing method in `SimulationContext.__init__` | Medium | `states.py` |
| 4 | Replace broad `except Exception:` with specific exceptions | Medium | 15+ files |

### Phase 2: Structural Improvements

| # | Task | Impact | Files |
|---|------|--------|-------|
| 5 | Move `load_area_and_waste_type_params` to `logic/src/utils/` | Medium | `loader.py`, 6+ policy files |
| 6 | Split `solutions.py` (1,518 lines) into 3 modules | Medium | `solutions.py` |
| 7 | Split `log_utils.py` (904 lines) into logging + visualization | Low | `log_utils.py` |
| 8 | Disambiguate policy naming between `policies/` and `models/policies/` | Medium | Multiple |

### Phase 3: Polish

| # | Task | Impact | Files |
|---|------|--------|-------|
| 9 | Document ~10 unexplained magic numbers | Low | Scattered |
| 10 | Standardize config loading documentation | Low | `config_loader.py` |
| 11 | Plan deprecation of backward compatibility shims | Low | `adapters.py` |
| 12 | Add inline comments to utility functions in `function.py` (644 lines) | Low | `function.py` |

---

## Part 4: Overall Assessment

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design Patterns | 9/10 | Exemplary Command, State, Factory, Mediator usage |
| Documentation | 8/10 | Comprehensive module/class docs; lighter in utilities |
| Type Safety | 8/10 | 90%+ coverage but `Any` overuse in context objects |
| Naming Conventions | 8.5/10 | Clear, descriptive; appropriate mathematical notation |
| Code Duplication | 4/10 | Policy layer has significant repetition |
| Error Handling | 5/10 | Too many broad `except Exception:` blocks |
| Module Organization | 7/10 | Clean boundaries but dual import paths confuse |
| File Size Management | 7/10 | 7 files >750 lines; largest at 1,518 |
| Configuration | 6/10 | Three competing patterns without clear hierarchy |
| **Overall** | **7.3/10** | **Strong architecture with targeted cleanup needed** |

The codebase is architecturally sound with clear domain modeling and professional documentation practices. The primary readability issues stem from code duplication in the policy layer, broad exception handling, and the dual import path structure - all of which are addressable without architectural changes.
