"""ZenML evaluation pipeline for WSmart-Route.

Wraps :func:`~logic.src.pipeline.features.eval.run_evaluate_model` in a
three-step ZenML pipeline:

1. **prepare_eval_config** — serialise the Hydra config to a plain dict.
2. **run_eval_step** — reconstruct the config, create a
   :class:`ZenMLBridge`, and delegate to :func:`run_evaluate_model`.
3. **log_eval_summary** — record a completion marker as a ZenML artifact.

The pipeline is invoked from
:func:`~logic.src.pipeline.features.eval._run_eval_via_zenml`
when ``cfg.tracking.zenml_enabled`` is ``True``.

Attributes:
    _ZENML_AVAILABLE: Whether ZenML is available.
    prepare_eval_config: Step to prepare evaluation config.
    run_eval_step: Step to run evaluation.
    log_eval_summary: Step to log evaluation summary.
    _eval_pipeline: The evaluation pipeline.

Example:
    >>> config = Config()
    >>> eval_pipeline(config)
"""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import OmegaConf

import logic.src.pipeline.features.eval as eval_module
from logic.src.configs import Config
from logic.src.tracking.integrations.zenml_bridge import ZenMLBridge
from logic.src.tracking.logging.pylogger import get_pylogger

try:
    from zenml import pipeline as zenml_pipeline  # type: ignore[import-not-found]
    from zenml import step  # type: ignore[import-not-found]

    _ZENML_AVAILABLE = True
except ImportError:
    _ZENML_AVAILABLE = False
    zenml_pipeline = None  # type: ignore[assignment]
    step = None  # type: ignore[assignment]

logger = get_pylogger(__name__)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

if _ZENML_AVAILABLE:

    @step  # type: ignore[misc]
    def prepare_eval_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through: makes the serialised config a ZenML artifact.

        Args:
            config_dict: Configuration dict.

        Returns:
            Configuration dict.
        """
        return config_dict

    @step(experiment_tracker="mlflow_tracker")  # type: ignore[misc]
    def run_eval_step(config_dict: Dict[str, Any]) -> str:
        """Execute evaluation inside a ZenML-managed MLflow run.

        The ``experiment_tracker`` decorator causes ZenML to start an MLflow
        run before entering this step and end it on exit.  A
        :class:`~logic.src.tracking.integrations.zenml_bridge.ZenMLBridge`
        is attached to the WSTracker run so metrics/params are dual-written.

        Args:
            config_dict: Configuration dict.

        Returns:
            Status string.
        """
        cfg = OmegaConf.structured(Config)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(config_dict))
        cfg = OmegaConf.to_object(cfg)
        assert isinstance(cfg, Config)

        bridge = ZenMLBridge()
        eval_module.run_evaluate_model(cfg, sinks=[bridge])
        return "completed"

    @step  # type: ignore[misc]
    def log_eval_summary(status: str) -> str:
        """Record evaluation completion as a ZenML artifact.

        Args:
            status: Status string.

        Returns:
            Status string.
        """
        logger.info(f"Evaluation pipeline {status}")
        return status

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------

    @zenml_pipeline(name="wsmart_route_evaluation")  # type: ignore[misc]
    def _eval_pipeline(config_dict: Dict[str, Any]) -> str:
        """Three-step evaluation pipeline.

        Args:
            config_dict: Configuration dict.

        Returns:
            Status string.
        """
        cfg_art = prepare_eval_config(config_dict)
        status = run_eval_step(cfg_art)
        return log_eval_summary(status)


# ---------------------------------------------------------------------------
# Public entry point (called from eval/__init__.py)
# ---------------------------------------------------------------------------


def eval_pipeline(cfg: Any) -> None:
    """Serialise *cfg* and launch the ZenML evaluation pipeline.

    Args:
        cfg: Root Hydra :class:`~logic.src.configs.Config` object.

    Raises:
        ImportError: If ZenML is not installed.
    """
    if not _ZENML_AVAILABLE:
        raise ImportError("zenml is not installed — cannot run ZenML evaluation pipeline")

    config_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    _eval_pipeline(config_dict=config_dict)
