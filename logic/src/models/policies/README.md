# RL Training Policy Wrappers

This directory contains **RL training wrappers** for classical optimization algorithms.

## Purpose

These wrappers make classical solvers compatible with the PyTorch-based RL training pipeline. They inherit from `ConstructivePolicy` and are used for:

- Generating expert demonstrations (imitation learning)
- Providing baselines during RL training
- Vectorized batch execution during training

## Key Files

| File | Description |
|------|-------------|
| `classical/alns.py` | `VectorizedALNS` - Batched ALNS for training |
| `classical/hgs.py` | `VectorizedHGS` - Batched HGS for training |
| `classical/local_search/` | Local search operators (2-opt, Or-opt, etc.) |

## Related Directory

See `logic/src/policies/` for **simulation-facing adapters** (e.g., `policy_alns.py`, `policy_hgs.py`) used during simulation experiments.
