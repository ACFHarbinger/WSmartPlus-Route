"""Training engine for WSmart-Route.

Attributes:
    run_training: Run single model training.
    _run_training_via_zenml: Run training through the ZenML pipeline.
    _build_experiment_name: Derive a human-readable experiment name from the Hydra config.
    _build_callbacks: Instantiate training callbacks from Hydra config.
    _log_training_params: Log training configuration parameters.
    _track_val_dataset: Track the validation dataset if available.

Example:
    >>> from logic.src.pipeline.features.train.engine import run_training
    >>> run_training(cfg)
    200000.0
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any, List, Optional, cast

import hydra
import omegaconf
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger

import logic.src.pipeline.features.train.zenml_train_pipeline as zenml_train_pipeline_module
import logic.src.tracking as wst
from logic.src.configs import Config
from logic.src.interfaces import ITraversable
from logic.src.pipeline.callbacks import SpeedMonitor
from logic.src.pipeline.features.train.model_factory.builder import create_model
from logic.src.pipeline.rl.common.trainer import WSTrainer
from logic.src.tracking.logging.pylogger import get_pylogger

try:
    from logic.src.tracking.integrations.zenml_bridge import configure_zenml_stack
except ImportError:
    configure_zenml_stack = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from logic.src.pipeline.rl.common.base.module import RL4COLitModule

logger = get_pylogger(__name__)


def _build_experiment_name(cfg: Config) -> str:
    """Derive a human-readable experiment name from the Hydra config.

    Args:
        cfg: Root configuration object.

    Returns:
        Human-readable experiment name string.
    """
    parts = [
        getattr(cfg.env, "name", "env"),
        str(getattr(cfg.env, "num_loc", "")),
        getattr(cfg.model, "name", getattr(cfg, "model_name", "")),
        getattr(cfg.rl, "algorithm", ""),
    ]
    return "-".join(p for p in parts if p)


def _build_callbacks(cfg: Config) -> list:
    """Instantiate training callbacks from Hydra config.

    Handles ``ITraversable`` configs (direct ``_target_`` dicts and nested
    name-to-config dicts), links ``CleanProgressBar`` ↔ ``TerminalChartCallback``
    when both are present, and always includes a :class:`SpeedMonitor`.

    Args:
        cfg: Root configuration object.

    Returns:
        List of instantiated callback objects.
    """
    callbacks: list = [SpeedMonitor(epoch_time=True)]
    if cfg.train.callbacks:
        for cb_cfg in cfg.train.callbacks:
            cb_cfg_obj: object = cb_cfg
            if isinstance(cb_cfg_obj, ITraversable):
                if "_target_" in cb_cfg_obj:
                    callbacks.append(hydra.utils.instantiate(cb_cfg_obj))
                else:
                    for _, actual_cfg in cb_cfg_obj.items():
                        actual_cfg_obj: object = actual_cfg
                        if isinstance(actual_cfg_obj, ITraversable) and "_target_" in actual_cfg_obj:
                            callbacks.append(hydra.utils.instantiate(actual_cfg_obj))
            else:
                with contextlib.suppress(Exception):
                    callbacks.append(hydra.utils.instantiate(cb_cfg))

    progress_bar = next((c for c in callbacks if c.__class__.__name__ == "CleanProgressBar"), None)
    terminal_chart = next((c for c in callbacks if c.__class__.__name__ == "TerminalChartCallback"), None)
    if progress_bar is not None and terminal_chart is not None:
        progress_bar.set_chart_callback(terminal_chart)  # type: ignore[attr-defined]

    return callbacks


def run_training(cfg: Config, sinks: Optional[List[Any]] = None) -> float:
    """Run single model training.

    Args:
        cfg: Root Hydra configuration object.
        sinks: Optional list of tracking sinks (e.g. :class:`ZenMLBridge`)
            to attach to the WSTracker run. When ``None`` (the default),
            :class:`MLflowBridge` is auto-attached if
            ``cfg.tracking.mlflow_enabled`` is ``True``.

    Returns:
        Validation reward from the best epoch.
    """
    # ----- ZenML dispatch (opt-in) -----
    tracking = getattr(cfg, "tracking", None)
    zenml_enabled = bool(getattr(tracking, "zenml_enabled", False))
    if zenml_enabled and sinks is None:
        # First invocation from CLI / main — delegate to ZenML pipeline
        return _run_training_via_zenml(cfg)

    seed_everything(cfg.seed)

    # Enable Tensor Core acceleration for Ampere+ GPUs
    if torch.cuda.is_available() and cfg.train.precision in ["16-mixed", "bf16-mixed"]:
        torch.set_float32_matmul_precision("medium")

    model_raw = create_model(cfg)
    model = cast("RL4COLitModule", model_raw)

    callbacks = _build_callbacks(cfg)

    trainer = WSTrainer(
        max_epochs=cfg.train.n_epochs,
        project_name="wsmart-route",
        experiment_name=cfg.experiment_name,
        accelerator=cfg.device if cfg.device != "cuda" else "auto",
        devices=cfg.train.devices,
        strategy=cfg.train.strategy,
        gradient_clip_val=(float(cfg.rl.max_grad_norm) if cfg.rl.algorithm != "ppo" else 0.0),
        logger=CSVLogger(cfg.tracking.log_dir or "logs", name=""),
        callbacks=callbacks,  # type: ignore[arg-type]
        precision=cfg.train.precision,
        log_every_n_steps=cfg.tracking.log_step,
        model_weights_path=cfg.train.model_weights_path,
        logs_dir=cfg.tracking.log_dir,
        reload_dataloaders_every_n_epochs=cfg.train.reload_dataloaders_every_n_epochs,
        enable_progress_bar=False,
        tracking_cfg=cfg.tracking,
    )

    # --- Centralised experiment tracking ---
    experiment_name = cfg.experiment_name or _build_experiment_name(cfg)
    tracker = wst.init(experiment_name=experiment_name)
    run_tags = {
        "algorithm": str(getattr(cfg.rl, "algorithm", "")),
        "model": str(getattr(cfg.model, "name", getattr(cfg, "model_name", ""))),
        "problem": str(getattr(cfg.env, "name", "")),
        "num_loc": str(getattr(cfg.env, "num_loc", "")),
        "seed": str(cfg.seed),
    }
    run = tracker.start_run(experiment_name, run_type="training", tags=run_tags)
    run.__enter__()

    # ----- Attach secondary sinks -----
    if sinks is not None:
        for sink in sinks:
            run.add_sink(sink)
    else:
        # Auto-attach MLflowBridge when running outside ZenML
        mlflow_enabled = bool(getattr(tracking, "mlflow_enabled", False))
        if mlflow_enabled:
            with contextlib.suppress(Exception):
                wst.MLflowBridge.attach(
                    run,
                    mlflow_tracking_uri=str(getattr(tracking, "mlflow_tracking_uri", "mlruns")),
                    experiment_name=str(getattr(tracking, "mlflow_experiment_name", "wsmart-route")),
                    run_name=run.run_id[:8],
                    tags=run_tags,
                )

    try:
        # Log training configuration as params
        _log_training_params(run, cfg)

        # Track validation dataset if available
        val_ds = cfg.train.val_dataset
        if val_ds and os.path.exists(str(val_ds)):
            run.log_dataset_event(
                "load",
                file_path=str(val_ds),
                metadata={
                    "variable_name": "val_ds",
                    "source_file": "features/train/engine.py",
                    "source_line": 151,
                },
            )
            run.watch_file(str(val_ds))

        trainer.fit(model)

        # Save final weights if path is provided
        if cfg.train.final_model_path:
            logger.info("Saving final model weights to: %s", cfg.train.final_model_path)
            model.save_weights(cfg.train.final_model_path)
            logger.info("Final model weights saved successfully.")
            run.log_artifact(cfg.train.final_model_path, artifact_type="model")
        else:
            logger.info("No final_model_path provided, skipping weight saving.")

        val_reward = trainer.callback_metrics.get("val/reward", torch.tensor(0.0)).item()
        run.log_metric("best/val_reward", val_reward)
        run.__exit__(None, None, None)
        return val_reward

    except Exception as exc:
        run.__exit__(type(exc), exc, exc.__traceback__)
        raise


def _log_training_params(run: wst.Run, cfg: Config) -> None:
    """Flatten and log relevant config sections as run parameters.

    Args:
        run: WSTracker run object.
        cfg: Root configuration object.
    """
    sections = {}
    for attr in ("train", "rl", "env", "model"):
        section = getattr(cfg, attr, None)
        if section is None:
            continue
        try:
            sections[attr] = omegaconf.OmegaConf.to_container(section, resolve=True)
        except Exception:
            with contextlib.suppress(Exception):
                sections[attr] = {k: getattr(section, k) for k in dir(section) if not k.startswith("_")}

    # Always include top-level seed / experiment_name
    sections["seed"] = cfg.seed
    sections["experiment_name"] = getattr(cfg, "experiment_name", "")
    run.log_params(sections)


# ---------------------------------------------------------------------------
# ZenML dispatch
# ---------------------------------------------------------------------------


def _run_training_via_zenml(cfg: Config) -> float:
    """Dispatch training to the ZenML training pipeline.

    Called when ``cfg.tracking.zenml_enabled`` is ``True`` and no external
    sinks were injected (i.e. the call originates from the CLI, not from
    inside a ZenML step).

    Args:
        cfg: Root configuration object.

    Returns:
        Best validation reward returned by the ZenML pipeline.
    """
    tracking = getattr(cfg, "tracking", None)
    mlflow_uri = str(getattr(tracking, "mlflow_tracking_uri", "mlruns"))
    stack_name = str(getattr(tracking, "zenml_stack_name", "wsmart-route-stack"))

    # Ensure the ZenML stack is configured before launching the pipeline
    if configure_zenml_stack is None or not configure_zenml_stack(mlflow_uri, stack_name=stack_name):
        logger.warning("ZenML stack configuration failed — falling back to direct training.")
        return run_training(cfg, sinks=[])
    try:
        result = zenml_train_pipeline_module.training_pipeline(cfg)
        return result if isinstance(result, float) else 0.0
    except Exception as exc:
        logger.warning(f"ZenML training pipeline failed — falling back to direct training: {exc}")
        return run_training(cfg, sinks=[])
