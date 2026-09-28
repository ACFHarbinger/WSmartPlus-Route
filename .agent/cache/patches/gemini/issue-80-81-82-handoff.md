# Gemini — Neural Agent & Attention Model Clean-Up (#80, #81, #82)

**Date**: 2026-09-28  
**Base commit**: `6d500ed02` on `main`  
**Worktree**: `~/.cache/wsr-review/gemini-code` (isolated; shared checkout was not modified; `assets/papers/**` untouched)  
**Python Environment**: `~/.cache/wsr-main-venv/bin/python`

---

## 1. Patches Delivered

All patches reside in `.agent/cache/patches/gemini/`:

1. **`issue-80-neural-agent-robustness.patch`**:
   - Fixes **B-gemini-01**: `AttentionModelPolicy` now forwards `norm_config=NormalizationConfig(norm_type=normalization)` to `GraphAttentionEncoder`, ensuring `normalization='layer'` creates `LayerNorm` rather than silently defaulting to `BatchNorm1d`. `GraphAttentionEncoder.__init__` accepts `normalization: Optional[str] = None` as well.
   - Fixes **B-gemini-02**: `SimulationMixin.compute_simulator_day` in `neural_agent/simulation.py:76` returns `([0, 0], 0, ...)` instead of `([0], 0, ...)` on empty mandatory set, adhering strictly to Owner Ruling D3.
   - Fixes **B-gemini-03**: `NeuralAgentPolicy.execute` in `policy_na.py:144-157` now computes physical collection in kilograms: `(real_c / 100.0) * volume * density * revenue_kg` using ground-truth `real_c` (fallback `c`), correcting both the percent-vs-kg scaling and noisy sensor reading.
   - Adds regression tests in `logic/test/unit/policies/test_na_robustness.py`.

2. **`issue-81-dead-layers.patch`**:
   - Resolves **D-gemini-01** and **B-gemini-04/05**: Removed dead projection layer `self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)` in `GlimpseDecoder` (`logic/src/models/subnets/decoders/glimpse/decoder.py`). Precompute sets `graph_context=None`.
   - Updated `AttentionDecoderCache` (`logic/src/models/subnets/decoders/common/cache.py`) to support `graph_context: Optional[torch.Tensor] = None`, guarding slicing operations.
   - Guarded `MatNetDecoder.forward` (`logic/src/models/subnets/decoders/matnet/decoder.py`) against `fixed.graph_context is None`.
   - Maintained checkpoint compatibility in `logic/src/utils/model/loader.py`: expanded `_UNUSED_LEGACY_KEYS` with `"decoder.project_fixed_context."` and `"project_fixed_context."`, and filtered them from both `missing_keys` and `unexpected_keys`.
   - Adds unit test `logic/test/unit/models/test_checkpoint_compatibility.py` verifying that legacy checkpoints containing `decoder.project_fixed_context.weight` load cleanly while unexpected unknown keys still trigger errors.

3. **`issue-82-neural-refactors.patch`**:
   - Executes **M-gemini-02**: Deduplicated the ~90-line autoregressive decoding loop in `DeepDecoderPolicy` (`logic/src/models/core/attention_model/deep_decoder_policy.py`). `AttentionModelPolicy.forward` in `policy.py` now flattens 3D logits (`if logits.dim() == 3: logits = logits[:, 0, :] if logits.size(1) > 1 else logits.squeeze(1)`), allowing `DeepDecoderPolicy` to directly inherit `AttentionModelPolicy` without reimplementing `forward`.
   - Executes **M-cursor-01 (NA part)**: `NeuralAgentPolicy` in `policy_na.py` now implements `_get_config_key() -> str` (`"na"`), `_config_class() -> Optional[Type]` (`NeuralParams`), and reuses `_validate_mandatory(mandatory)` for early exit on empty mandatory sets.
   - Adds unit test `test_deep_decoder_policy_forward` in `logic/test/unit/models/subnets/test_deep_decoder.py`.

---

## 2. Verification & Test Evidence

### 2.1 Patch Application Checks
All three patches check cleanly against base `6d500ed02` independently and in sequence:
```bash
# Independent clean check:
git apply --check .agent/cache/patches/gemini/issue-80-neural-agent-robustness.patch  # OK
git apply --check .agent/cache/patches/gemini/issue-81-dead-layers.patch               # OK
git apply --check .agent/cache/patches/gemini/issue-82-neural-refactors.patch         # OK

# Sequential application:
git apply .agent/cache/patches/gemini/issue-80-neural-agent-robustness.patch
git apply .agent/cache/patches/gemini/issue-81-dead-layers.patch
git apply .agent/cache/patches/gemini/issue-82-neural-refactors.patch
```

### 2.2 Compilation and Import Sweep
```bash
python -m compileall -q logic                                        # Exited 0
flock ~/.cache/wsr-review/heavy.lock python .agent/cache/tools/import_sweep.py # 2107 modules checked; 0 core failures
```

### 2.3 Unit & Regression Test Suite
Executed using `~/.cache/wsr-main-venv/bin/pytest`:
- `logic/test/unit/policies/test_na_robustness.py`: 3 passed (B-gemini-01, B-gemini-02, B-gemini-03 regression tests)
- `logic/test/unit/models/test_checkpoint_compatibility.py`: 2 passed (D-gemini-01 attribute check & pre-change checkpoint loading)
- `logic/test/unit/models/subnets/test_deep_decoder.py`: 5 passed (DeepGATDecoder and DeepDecoderPolicy forward loop)
- `logic/test/unit/models/test_models.py`: 19 passed
- `logic/test/unit/utils/model/test_load_model.py`: 1 passed
- `logic/test/unit/models/test_matnet_parity.py`: 4 passed
- `logic/test/unit/models/`: 164 passed, 3 skipped

---

## 3. Behavior Changes

1. **AttentionModel Normalization**: YAML configurations specifying `normalization: layer` now correctly construct `LayerNorm` instead of silently defaulting to `BatchNorm1d`.
2. **Neural Agent Empty Tour**: When mandatory bins are present in configuration but evaluate to an empty set, `NeuralAgent` returns `[0, 0]` instead of `[0]`.
3. **Neural Agent Revenue Accounting**: Revenue collected by `NeuralAgentPolicy` now scales by physical kilograms `(real_c / 100.0) * volume * density * revenue_kg` using `bins.real_c` instead of directly multiplying percentage fill from `bins.c`.
4. **Checkpoint Loading**: Checkpoints generated prior to this round containing `decoder.project_fixed_context.weight` are safely filtered as legacy unused weights during `load_model()`.

---

## 4. M-gemini-01 Migration Plan (Unify `AttentionModel` with `AttentionModelPolicy`)

Per the brief, M-gemini-01 is skipped for this clean-up round to prevent disruption while other lanes finalize their runs. Below is the technical plan of record for unifying the two classes:

### Problem Statement
Currently, the codebase contains two parallel Attention Model architectures:
1. **Legacy `AttentionModel`** (`logic/src/models/core/attention_model/model.py`):
   - Implements PyTorch `nn.Module` with `DecodingMixin`.
   - Utilizes `component_factory` (`AttentionComponentFactory`).
   - Supports legacy parameters: `pomo_size`, `shrink_size`, `predictor_layers`, `decoder_type`, `temporal_horizon`.
   - Input format: nested dictionary `{"loc": ..., "demand": ...}`.
2. **RL4CO `AttentionModelPolicy`** (`logic/src/models/core/attention_model/policy.py`):
   - Inherits `AutoregressivePolicy` (`logic/src/models/common/autoregressive/policy.py`).
   - Uses `TensorDict` inputs compatible with RL4CO environments and PyTorch Lightning.
   - Clean, modular encoder/decoder structure.

### Proposed Architecture & Steps
1. **Phase 1: Shared Core Subnets (Completed)**
   - Both models now share `GraphAttentionEncoder` and `GlimpseDecoder`.
   - The encoder output parity has been verified (`diff = 0.0`).
2. **Phase 2: Adapter Shim in `AttentionModel`**
   - Refactor `AttentionModel` to wrap `AttentionModelPolicy` internally:
     - Wrap input `dict` into `TensorDict` via a lightweight utility `dict_to_tensordict(input)`.
     - Delegate `forward` and `_precompute` to `self.policy.forward` and `self.policy.decoder._precompute`.
     - Retain `component_factory` as an optional argument for backward compatibility.
3. **Phase 3: State Dict Remapping & Checkpoint Migration**
   - Provide an automatic key remapper in `loader.py`:
     - `context_embedder.init_embed.*` $\leftrightarrow$ `init_embedding.node_embed.*`
     - `context_embedder.init_embed_depot.*` $\leftrightarrow$ `init_embedding.depot_embed.*`
   - Run a migration utility over existing `.pt` weights in `assets/weights/` to convert state dicts into canonical `policy.*` keys.
4. **Phase 4: Deprecation and Deletion**
   - Deprecate `AttentionModel` with a `warnings.warn("Use AttentionModelPolicy instead", DeprecationWarning)`.
   - After one release cycle, replace direct imports of `AttentionModel` with aliases to `AttentionModelPolicy`.
