# Architectural Analysis Prompt

**Intent:** Use Chain-of-Thought reasoning to explore the Python/Rust boundary.

## The Prompt

I need to understand the interface between the high-performance Rust policies and the Python logic layer.

Using **Chain-of-Thought reasoning**, analyze the relationship between:
- The Rust bindings in `policy/src/lib.rs` (specifically the solve and solve_batch functions).
- The Python wrapper in `logic/src/policies/vrpp_optimizer.py` (or relevant adapter).
- The data structures passed for `dist_matrix`, `demands`, and `coords`.

Explain potential bottlenecks in data marshalling between Python/Rust and suggest if `PyO3` usage in `lib.rs` is optimized for zero-copy memory access based on the provided code.
