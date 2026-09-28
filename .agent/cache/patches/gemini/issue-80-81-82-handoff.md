# Gemini — Neural Agent & Attention Model Clean-Up (#80, #81, #82 Revision)

**Date**: 2026-09-28  
**Base commit**: `f1975cdf9` on `main` (clean-up batch 1 landed)  
**Worktree**: `~/.cache/wsr-review/gemini-code` (isolated; shared checkout was not modified; `assets/papers/**` untouched)  
**Python Environment**: `~/.cache/wsr-main-venv/bin/python`

---

## 1. Patches Manifest & SHA-256 Hashes

All delivered patches reside in `.agent/cache/patches/gemini/`:

| Issue | Patch File | SHA-256 | Status |
|---|---|---|---|
| **#80** (C3 Robustness) | `issue-80-neural-agent-robustness.patch` | `a467d4e66c71c4fa82c256c653427877ab5c8554908c0f57aecd5afeb198a573` | **Delivered & Verified** (with 3 regression tests) |
| **#81** (C4 Dead Code) | `issue-81-dead-layers.patch` | `b99de27c2e55f0e4efa2377656fc205a9ab75e5443ac4bc5843bfce398893486` | **Landed in Batch 1** (`159099b56`) |
| **#82** (C5 Refactors) | `issue-82-neural-refactors.patch` | `66d4a7b101905af2380b666d119ca71a61f653a951a88f9b9cffb413b8878177` | **Delivered & Verified** (tensor-mask crash fixed, 4 regression tests, M-gemini-03 deferred) |

---

## 2. Details of Changes

### 2.1 Issue #80: Neural Agent Robustness Fixes (`issue-80-neural-agent-robustness.patch`)

1. **B-gemini-01 (Normalization parameter forwarding)**:
   - `AttentionModelPolicy` (`logic/src/models/core/attention_model/policy.py`) now forwards `norm_config=NormalizationConfig(norm_type=normalization)` to `GraphAttentionEncoder`, ensuring `normalization='layer'` creates `LayerNorm` rather than silently defaulting to `BatchNorm1d`.
   - `GraphAttentionEncoder.__init__` (`logic/src/models/subnets/encoders/gat/encoder.py`) accepts `normalization: Optional[str] = None` and converts it to `NormalizationConfig`.
2. **B-gemini-02 (Empty tour depot loop)**:
   - `SimulationMixin.compute_simulator_day` in `logic/src/policies/route_construction/learning_algorithms/neural_agent/simulation.py:76` returns `([0, 0], 0, {"attention_weights": torch.tensor([]), "graph_masks": [], "mandatory_empty": True})` instead of `([0], 0, ...)` on empty mandatory set, adhering strictly to Owner Ruling D3.
3. **B-gemini-03 (Revenue unit scaling and ground-truth values)**:
   - `NeuralAgentPolicy.execute` in `logic/src/policies/route_construction/learning_algorithms/neural_agent/policy_na.py:144-157` now computes physical collection in kilograms: `(real_c / 100.0) * volume * density * revenue_kg` using ground-truth `real_c` (fallback `c`), correcting both the percent-vs-kg scaling and noisy sensor reading.
4. **Regression Tests**:
   - Added `logic/test/unit/policies/test_na_robustness.py`:
     - `test_b_gemini_01_normalization_forwarded_to_gat_encoder`: verifies `normalization='layer'` constructs `LayerNorm` and `'batch'` constructs `BatchNorm1d`.
     - `test_b_gemini_02_empty_mandatory_returns_depot_loop`: verifies `compute_simulator_day` returns `[0, 0]` with metadata flag `mandatory_empty=True`.
     - `test_b_gemini_03_revenue_units_and_ground_truth`: verifies revenue is computed on `real_c` converted from percentage to kg.

### 2.2 Issue #81: Dead Projection Layers (`issue-81-dead-layers.patch`)
- **Status**: Landed by Claude into `main` at `159099b56` during Batch 1.
- Removed dead projection layer `self.project_fixed_context` in `GlimpseDecoder`.
- Checkpoints generated before the change load safely with filtered legacy keys.

### 2.3 Issue #82: Neural Refactors & Tensor-Mask Crash Fix (`issue-82-neural-refactors.patch`)

1. **Fix for Codex Finding §10.2 #1 (Tensor Mandatory Masks Crash)**:
   - Previously, `NeuralAgentPolicy.execute` called `BaseRoutingPolicy._validate_mandatory`, whose `if not mandatory` threw `RuntimeError: Boolean value of Tensor with more than one value is ambiguous. Use a.any() or a.all()` when passed a multi-element boolean tensor (e.g. `torch.tensor([[False, True, False]])`).
   - Overrode `_validate_mandatory(self, mandatory: Any)` in `NeuralAgentPolicy` to safely handle `None`, lists, tuples, sets, integer ID tensors, and 1D/2D boolean tensor masks.
   - For boolean tensors: checks `mandatory.numel() == 0` or `not mandatory.any()` to detect empty mandatory constraints without ambiguous truth evaluations.
   - In `execute()`: empty mandatory inputs trigger an early exit with `([0, 0], 0.0, 0.0, search_context, multi_day_context)` before accessing the model, while non-empty masks proceed cleanly to model unpacking and execution.
2. **M-cursor-01 (NA Part)**:
   - Implemented `_config_class(cls) -> Optional[Type[Any]]` returning `NeuralParams`.
   - Implemented `_get_config_key(self) -> str` returning `"na"`.
   - Initialized `NeuralParams` from typed config or `config["na"]` dictionary.
3. **M-gemini-02 (Deduplicate Autoregressive Decoding Loop)**:
   - `DeepDecoderPolicy` (`logic/src/models/core/attention_model/deep_decoder_policy.py`) now directly subclasses `AttentionModelPolicy`, eliminating the duplicate ~90-line sequential construction loop.
   - In `AttentionModelPolicy.forward` (`logic/src/models/core/attention_model/policy.py`), added 3D logits flattening (`if logits.dim() == 3: logits = logits[:, 0, :] if logits.size(1) > 1 else logits.squeeze(1)`), ensuring compatibility with multi-head decoders such as `DeepGATDecoder`.
4. **Regression Tests**:
   - In `logic/test/unit/policies/test_na_robustness.py`:
     - `test_validate_mandatory_empty_and_nonempty_cases`: tests `None`, empty/nonempty lists, tuples, sets, integer tensors, and 1D/2D boolean tensor masks.
     - `test_execute_multi_element_tensor_mask_no_ambiguous_boolean_crash`: tests the exact scenario from Codex reproducer (`mask = torch.tensor([[False, True, False]])`) and confirms no ambiguous truth-value error is raised.
     - `test_execute_empty_tensor_mask_early_exit`: tests that calling `execute(mandatory=torch.tensor([[False, False, False]]))` or `execute(mandatory=[])` cleanly returns `[0, 0]` without requiring model context kwargs.
   - In `logic/test/unit/models/subnets/test_deep_decoder.py`:
     - `test_deep_decoder_policy_forward`: verifies that `DeepDecoderPolicy(env_name="vrpp", ...)` runs forward construction via inherited `AttentionModelPolicy` and returns valid actions, reward, and log likelihood.

---

## 3. Explicit Deferrals

### 3.1 M-gemini-01 (Unify Legacy `AttentionModel` with `AttentionModelPolicy`)
- **Status**: Deferred per brief instructions.
- **Migration Plan**:
  1. Both models already share `GraphAttentionEncoder` and `GlimpseDecoder`.
  2. Legacy `model.py` additionally exposes `component_factory`, `pomo_size`, `shrink_size`, `predictor_layers`, `decoder_type`, and `temporal_horizon`, requiring an adapter shim before full deletion.
  3. Migration to be executed in a dedicated model consolidation PR after test sweeps stabilize.

### 3.2 M-gemini-03 (Merge `VRPPInitEmbedding` and `VRPPContextEmbedder`)
- **Status**: Explicitly deferred.
- **Rationale**:
  - `VRPPInitEmbedding` (`logic/src/models/subnets/embeddings/vrpp.py:36-64`) is a static input feature projector: projects coordinates (`nn.Linear(2)`) and demand/waste values (`nn.Linear(1)` / `nn.Linear(2)`), replacing node 0 with depot features.
  - `VRPPContextEmbedder` (`logic/src/models/subnets/context/vrpp.py:44-58,100-125`) is an active step context projector: manages temporal horizons, concatenated depot detection, dynamic step state, and context projection into attention query space.
  - The two embedders share problem domain context but implement completely different interface contracts, input dimensions, and lifecycle behaviors. Unifying them offers minimal code savings (~45 LOC) while introducing high regression risk across constructive decoders.

---

## 4. Verification & Test Evidence

### 4.1 Git Apply Checks on Base `f1975cdf9`
Both patches apply cleanly in sequence to clean `f1975cdf9`:
```bash
git apply --check .agent/cache/patches/gemini/issue-80-neural-agent-robustness.patch # OK
git apply .agent/cache/patches/gemini/issue-80-neural-agent-robustness.patch       # Applied
git apply --check .agent/cache/patches/gemini/issue-82-neural-refactors.patch       # OK
git apply .agent/cache/patches/gemini/issue-82-neural-refactors.patch             # Applied
```

### 4.2 Test Suite Execution
Executed with `~/.cache/wsr-main-venv/bin/pytest`:
```bash
$ pytest logic/test/unit/policies/test_na_robustness.py logic/test/unit/models/subnets/test_deep_decoder.py
============================== 11 passed in 0.16s ==============================
```
Tests passed:
- `test_b_gemini_01_normalization_forwarded_to_gat_encoder`: PASSED
- `test_b_gemini_02_empty_mandatory_returns_depot_loop`: PASSED
- `test_b_gemini_03_revenue_units_and_ground_truth`: PASSED
- `test_validate_mandatory_empty_and_nonempty_cases`: PASSED
- `test_execute_multi_element_tensor_mask_no_ambiguous_boolean_crash`: PASSED
- `test_execute_empty_tensor_mask_early_exit`: PASSED
- `test_deep_decoder_init`: PASSED
- `test_deep_decoder_precompute`: PASSED
- `test_deep_decoder_get_log_p`: PASSED
- `test_deep_decoder_slicing`: PASSED
- `test_deep_decoder_policy_forward`: PASSED

### 4.3 Direct Reproduction Verification (Codex Finding §10.2 #1)
Ran in-memory reproducer verifying:
- Non-empty boolean mask `torch.tensor([[False, True, False]])` passes validation cleanly without raising ambiguous tensor boolean error.
- Empty boolean mask `torch.tensor([[False, False, False]])` exits early with `([0, 0], 0.0, 0.0)`.
Result: **ALL CHECKS PASSED**.
