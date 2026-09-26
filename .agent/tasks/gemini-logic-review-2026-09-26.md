# Brief: Agy / Gemini (lane C: the Attention Model and the Neural Agent)

Read `.agent/tasks/logic-review-2026-09-26-common.md` first.

## Papers

- `bibliography/models/Attention_Model.pdf`: Kool, van Hoof & Welling, ICLR 2019.
  Your part is:
  - §3, the model: the encoder with N attention layers, batch normalisation,
    skip connections and an FF sublayer of width 512; the graph embedding as
    a mean; the context embedding; the glimpse with M = 8 heads; the single
    head with C = 10 tanh clipping; masking;
  - Appendix A, the VRP variants and how the problem's context is fed in.
- `bibliography/models/Pointer_Networks.pdf`, as background for the decoder
  only.

## Scope

- `logic/src/models/core/attention_model/**`:
  - `model.py`, `policy.py`, `decoding.py`;
  - `deep_decoder_policy.py` and `symnco_policy.py`, for duplication only.
- The subnets, embeddings and common modules the AM imports (follow imports
  from `policy.py`/`model.py`), plus `configs/models/*`.
- `utils/model/loader.py` (shared with lane A; coordinate on the bus).
- `policies/route_construction/learning_algorithms/neural_agent/**`:
  `agent.py`, `batch.py`, `simulation.py`, `policy_na.py`, `params.py`.

## Questions

1. **Paper fidelity**, per component:
   - the normalisation type and its placement;
   - the FF width, now `hidden_dim` (B-claude-04);
   - `sqrt(d_k)` scaling;
   - where the tanh clipping sits (logits, not the glimpse);
   - the VRPP context (remaining capacity, current node, and the profit
     features; check they are built from the right tensors);
   - the depot mask rule (no consecutive depot visits unless everything is
     done);
   - greedy vs sampling decoding.
2. **The two model classes.** The training `AttentionModelPolicy` and the
   legacy `AttentionModel`, which eval and NA use, both exist.
   - Do they compute the same function for the same weights?
     `.agent/cache/tools/loader_parity_check.py` exists; rerun it on `main`.
   - Can one of them be removed or reduced to a thin wrapper? That is an M row
     with a migration plan.
3. **The Neural Agent.**
   - Is the simulated state built with the same normalisation as training
     (units, the fill percentage vs a fraction)?
   - Is `must_go` enforced in the model mask or patched afterwards?
   - Is the returned tour depot-framed the same way as every other policy?
4. **Duplication.**
   - The attention and MHA implementations across `subnets`/`common`.
   - The several embedding classes that differ only in their input features.
   - The decoding strategies duplicated between `models/.../decoding.py` and
     `utils/decoding` (coordinate with lane A).
5. **Dead code.** Model code that no yaml or registry reaches. The other
   registered models are not dead.
