# Logging Architecture

This directory provides two logging systems, each chosen for its domain's requirements.

## `get_pylogger()` — Python `logging` (RL Training Pipeline)

**Used by**: `train.py`, `base.py`, `baselines.py`, `gat_lstm_manager.py`, `transforms.py`

Uses the standard Python `logging` module wrapped via `pylogger.py`. This is the correct
choice for the RL training pipeline because:

- **PyTorch Lightning compatibility**: Lightning's distributed training (DDP) integrates
  natively with Python `logging`, ensuring proper log rank filtering.
- **Hierarchical loggers**: Each module gets a named logger (`__name__`), enabling
  fine-grained log level control per subsystem.

## `loguru` — Structured Logging (Simulation & Evaluation Pipeline)

**Used by**: `eval.py`, `test.py`, `trainer.py`, `checkpoints.py`, `states.py`,
`actions.py`, `storage.py`, `analysis.py`

Uses the `loguru` library for its superior formatting and structured output. This is the
correct choice for the simulation pipeline because:

- **Rich formatting**: Simulation logs benefit from colored, timestamped output that
  `loguru` provides out of the box.
- **Structured logging**: The `structured_logging.py` module builds on `loguru` for
  JSON-formatted simulation metrics (see `log_utils.py`).
- **No DDP concerns**: Simulation runs are single-process, so Lightning compatibility
  is not required.

## When to Use Which

| Context | Logger | Import |
|---------|--------|--------|
| RL training modules (`pipeline/rl/`) | `get_pylogger` | `from logic.src.utils.logging.pylogger import get_pylogger` |
| Simulation modules (`pipeline/simulations/`) | `loguru` | `from loguru import logger` |
| Evaluation / testing (`pipeline/features/`) | `loguru` | `from loguru import logger` |
| Data transforms / preprocessing | `get_pylogger` | `from logic.src.utils.logging.pylogger import get_pylogger` |
