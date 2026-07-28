"""Model pipeline controller.

Entry point for all commands that operate on PyTorch neural models:
training (``train``, ``meta_train``, ``hpo``) and evaluation (``eval``).

Both groups share the same conceptual domain — building, fitting, and
assessing neural routing models — so they are co-located here rather than
spread across separate files.

Hydra drives configuration for all tasks in this module; the heavy logic
lives in :mod:`logic.src.pipeline.features.train` and
:mod:`logic.src.pipeline.features.eval`.

Example::

    python main.py train model=am env.name=vrpp env.num_loc=50
    python main.py meta_train experiment=meta_rl model=am
    python main.py hpo env.name=wcvrp hpo.n_trials=50
    python main.py eval eval.model_path=./weights/best.pt
"""

from typing import Any, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT_KEYS: List[str] = [
    "seed", "device", "experiment_name", "task",
    "output_dir", "run_name", "start", "tracking",
]

_TRAINING_TASKS = frozenset({"train", "meta_train", "hpo"})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_config(cfg: Any, label: str, filter_keys: Any = None) -> None:
    """Pretty-print a filtered view of a Hydra config.

    Args:
        cfg: The Hydra configuration object.
        label: Section header label (e.g. ``"TRAINING"``).
        filter_keys: Optional list of top-level keys to include.
    """
    from omegaconf import OmegaConf

    print("\n" + "=" * 80)
    print(f"HYDRA CONFIGURATION — {label}".center(80))
    print("=" * 80)
    display_cfg = OmegaConf.masked_copy(cfg, filter_keys) if filter_keys else cfg
    print(OmegaConf.to_yaml(display_cfg, resolve=False))
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Training main
# ---------------------------------------------------------------------------


def run_training(cfg: Any) -> float:
    """Dispatch a training, meta-RL, or HPO run.

    Delegates to :func:`~logic.src.pipeline.features.train.run_hpo` when
    ``cfg.hpo.n_trials > 0``, otherwise to
    :func:`~logic.src.pipeline.features.train.run_training`.

    Args:
        cfg: Hydra ``Config`` object (structured config).

    Returns:
        Scalar result value (loss, reward, or HPO best metric).
    """
    from logic.src.pipeline.features.train import run_hpo, run_training as _train

    if cfg.tracking.verbose:
        _print_config(
            cfg,
            "TRAINING",
            filter_keys=_ROOT_KEYS + ["env", "model", "train", "rl", "optim"],  # type: ignore[arg-type]
        )

    if cfg.hpo.n_trials > 0:
        return run_hpo(cfg)
    return _train(cfg)


# ---------------------------------------------------------------------------
# Evaluation main
# ---------------------------------------------------------------------------


def run_evaluation(cfg: Any) -> float:
    """Run model evaluation.

    Args:
        cfg: Hydra ``Config`` object (structured config).

    Returns:
        0.0 on success.
    """
    from logic.src.pipeline.features.eval import run_evaluate_model

    if cfg.tracking.verbose:
        _print_config(cfg, "EVALUATION", filter_keys=_ROOT_KEYS + ["eval"])  # type: ignore[arg-type]

    run_evaluate_model(cfg)
    return 0.0
