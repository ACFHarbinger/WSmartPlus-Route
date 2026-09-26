# Brief — Gemini / Agy (lane C: neural model stack)

Read `.agent/tasks/minimal-export-review-common.md` first. Bus: `.agent/bus/2026-09-25.md`.

## Lane C scope

- `logic/src/models/core/attention_model/` — `model.py` (legacy `AttentionModel`, used by eval and the Neural Agent), `policy.py` (`AttentionModelPolicy`, used by training), `deep_decoder_policy.py`, `symnco_policy.py`, `decoding.py`, `__init__.py`.
- `logic/src/models/subnets/` — `encoders/{common,gat}`, `decoders/{common,gat,glimpse}`, `factories/{attention,base}.py`, `modules/*` (12 files), `embeddings/**` (`static`, `dynamic`, `vrpp`, `context/*`, `state/*`, `edges/*`, `positional/*`).
- `logic/src/models/common/` — `autoregressive/`, `critic_network/`, `improvement/`, `non_autoregressive/`, `transductive/`.
- `logic/src/configs/models/*` and `logic/configs/models/{am,deep_decoder}.yaml`.
- Shared with lane A: `logic/src/utils/model/loader.py` — coordinate with Codex on the bus.

## Questions this lane must answer

1. **Two model classes, one checkpoint.** Training saves `AttentionModelPolicy` weights; eval and the simulator rebuild a legacy `AttentionModel` from `config.yaml` and load with `strict=False`. List every parameter name that does *not* match between the two (run both constructors with the same config and diff `state_dict().keys()`), every constructor argument the loader forwards, and any default that differs (normalisation, activation, `tanh_clipping`, `mask_inner`, `spatial_bias`, connection type). Each silent mismatch is a `major` bug. Recommend whether the legacy class can be deleted in favour of the policy class (removal row with the required loader change).
2. **Decoding.** `decoding.py` + `decoders/glimpse/decoder.py`: greedy/sampling/beam behaviour, `tanh_clipping`, masking of visited nodes and depot, handling of `mandatory` masks passed by the Neural Agent (`policies/route_construction/learning_algorithms/neural_agent/*` reads the model output — check the contract from the model side).
3. **Embeddings.** Which of `embeddings/{static,dynamic}.py`, `context/{base,generic,vrpp}.py`, `state/{env,vrpp}.py`, `edges/{base,none,tsp}.py`, `positional/*` are reachable for `vrpp` + `am`? R-claude-01 (positional) is yours; also check `dynamic.py` (was listed as MDAM/PolyNet-specific) and `edges/tsp.py`.
4. **`modules/`.** 12 files remain (`activation_function`, `connections`, `cross_attention`, `dynamic_hyper_connection`, `feed_forward`, `flash_attention`, `multi_head_attention`, `normalization`, `normalized_activation_function`, `skip_connection`, `static_hyper_connection`, ...). Which are reachable from the GAT encoder/glimpse decoder with the retained yaml (`connection_type: residual`)? Removal rows for the rest, with the `subnet_pruning.prunable_types.modules.always_keep` change.
5. **`models/common`.** R-claude-04 (`non_autoregressive`, `improvement`, `transductive`) is yours; also decide `critic_network` (only needed for `rl.baseline=critic`) and `deep_decoder_policy.py`/`symnco_policy.py` (DDAM/SymNCO variants — retained only because they sit in the AM package).
6. `configs/models/*.py` dataclasses: fields that no retained code reads (e.g. `n_predictor_layers`, `hyper_expansion`, `spatial_bias_scale`, MoE/temporal fields) → removal rows, mirrored in `am.yaml`/`train.yaml`.

Post claims and findings on the bus under `### Agy (Gemini) — 2026-09-25 (...)`.
