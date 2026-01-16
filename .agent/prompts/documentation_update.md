# Documentation Update Prompt

**Intent:** Keep Rust and Python components synchronized using a Refinement pattern.

## The Prompt

I have updated `policy/src/genetic.rs` to include a new crossover operator.

**Task:** Update the docstrings in `policy/src/lib.rs` to reflect that `solve_instance` now supports a `crossover_type` parameter (integer).

**Review Guidelines:**
1. Check consistency with PyO3 attributes `#[pyfunction]`.
2. Ensure the Rust types match the expected Python types in `logic/src/policies/`.
3. Generate the updated Rust function signature.
