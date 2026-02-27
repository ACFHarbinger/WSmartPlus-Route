"""ZenML training pipeline for WSmart-Route.

Wraps :func:`~logic.src.pipeline.features.train.engine.run_training` in a
three-step ZenML pipeline so every training run is version-tracked and
artifact-cached:

1. **prepare_training_config** — serialise the Hydra config to a plain dict.
2. **run_training_step** — reconstruct the config, create a
   :class:`ZenMLBridge`, and delegate to :func:`run_training`.
3. **log_training_summary** — record the best validation reward as a ZenML
   artifact.

The pipeline is invoked from
:func:`~logic.src.pipeline.features.train.engine._run_training_via_zenml`
when ``cfg.tracking.zenml_enabled`` is ``True``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict

from logic.src.tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)

# ---------------------------------------------------------------------------
# Lazy ZenML imports — guarded so the module can be imported even when the
# zenml package is not installed.
# ---------------------------------------------------------------------------

_ZENML_AVAILABLE = False
with contextlib.suppress(ImportError):
    from zenml import pipeline as zenml_pipeline  # type: ignore[import-not-found]
    from zenml import step  # type: ignore[import-not-found]

    _ZENML_AVAILABLE = True


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

if _ZENML_AVAILABLE:

    @step  # type: ignore[misc]
    def prepare_training_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through: makes the serialised config a ZenML artifact."""
        return config_dict

    @step(experiment_tracker="mlflow_tracker")  # type: ignore[misc]
    def run_training_step(config_dict: Dict[str, Any]) -> float:
        """Execute training inside a ZenML-managed MLflow run.

        The ``experiment_tracker`` decorator causes ZenML to start an MLflow
        run before entering this step and end it on exit.  A
        :class:`~logic.src.tracking.integrations.zenml_bridge.ZenMLBridge`
        is attached to the WSTracker run so metrics/params are dual-written.
        """
        from omegaconf import OmegaConf

        from logic.src.configs import Config
        from logic.src.pipeline.features.train.engine import run_training
        from logic.src.tracking.integrations.zenml_bridge import ZenMLBridge

        cfg = OmegaConf.structured(Config)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(config_dict))
        cfg = OmegaConf.to_object(cfg)
        assert isinstance(cfg, Config)

        bridge = ZenMLBridge()
        return run_training(cfg, sinks=[bridge])

    @step  # type: ignore[misc]
    def log_training_summary(val_reward: float) -> float:
        """Record the best validation reward as a ZenML artifact."""
        logger.info(f"Training complete — best val_reward: {val_reward:.4f}")
        return val_reward

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------

    @zenml_pipeline(name="wsmart_route_training")  # type: ignore[misc]
    def _training_pipeline(config_dict: Dict[str, Any]) -> float:
        """Three-step training pipeline."""
        cfg_art = prepare_training_config(config_dict)
        val_reward = run_training_step(cfg_art)
        return log_training_summary(val_reward)


# ---------------------------------------------------------------------------
# Public entry point (called from engine.py)
# ---------------------------------------------------------------------------


def training_pipeline(cfg: Any) -> float:
    """Serialise *cfg* and launch the ZenML training pipeline."""
    if not _ZENML_AVAILABLE:
        raise ImportError("zenml is not installed — cannot run ZenML training pipeline")

    from omegaconf import OmegaConf
    from zenml.client import Client

    config_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

    # 1. Launch the pipeline
    run_response = _training_pipeline(config_dict=config_dict)

    if run_response is None:
        return 0.0

    # 2. Extract the best validation reward from the run metadata
    # This assumes your pipeline logs a 'best_val_reward' in one of its steps
    try:
        run = Client().get_pipeline_run(run_response.id)

        # Access the specific step
        eval_step = run.steps["evaluator_step"]

        # Use .output to get the primary output artifact of the step
        # This is where the .load() method lives
        reward = eval_step.output.load()

        return float(reward)
    except (KeyError, AttributeError, ValueError):
        # Fallback if the artifact isn't a float or step is missing
        return 0.0
