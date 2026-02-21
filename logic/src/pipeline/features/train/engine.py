"""
Training engine for WSmart-Route.
"""

import contextlib
import os

import hydra
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger

import logic.src.tracking as wst
from logic.src.configs import Config
from logic.src.interfaces import ITraversable
from logic.src.pipeline.callbacks import SpeedMonitor
from logic.src.pipeline.features.train.model_factory.builder import create_model
from logic.src.pipeline.rl.common.trainer import WSTrainer


def _build_experiment_name(cfg: Config) -> str:
    """Derive a human-readable experiment name from the Hydra config."""
    parts = [
        getattr(cfg.env, "name", "env"),
        str(getattr(cfg.env, "num_loc", "")),
        getattr(cfg.model, "name", getattr(cfg, "model_name", "")),
        getattr(cfg.rl, "algorithm", ""),
    ]
    return "-".join(p for p in parts if p)


def run_training(cfg: Config) -> float:
    """Run single model training."""
    seed_everything(cfg.seed)

    # Enable Tensor Core acceleration for Ampere+ GPUs
    if torch.cuda.is_available() and cfg.train.precision in ["16-mixed", "bf16-mixed"]:
        torch.set_float32_matmul_precision("medium")

    model = create_model(cfg)

    # Setup callbacks
    callbacks = [SpeedMonitor(epoch_time=True)]
    if cfg.train.callbacks:
        for cb_cfg in cfg.train.callbacks:
            # Use ITraversable protocol for config-like objects
            if isinstance(cb_cfg, ITraversable):
                # Handle direct target dicts: {_target_: ...}
                if "_target_" in cb_cfg:
                    callbacks.append(hydra.utils.instantiate(cb_cfg))
                # Handle named dicts: {name: {_target_: ...}}
                else:
                    for _, actual_cfg in cb_cfg.items():
                        if isinstance(actual_cfg, ITraversable) and "_target_" in actual_cfg:
                            callbacks.append(hydra.utils.instantiate(actual_cfg))
            else:
                # Fallback for other types (e.g. if it's already an object or unhandled type)
                with contextlib.suppress(Exception):
                    callbacks.append(hydra.utils.instantiate(cb_cfg))

    # Link Progress Bar and Chart if both exist
    # This is necessary because they are instantiated separately by Hydra
    progress_bar = next((c for c in callbacks if c.__class__.__name__ == "CleanProgressBar"), None)
    terminal_chart = next((c for c in callbacks if c.__class__.__name__ == "TerminalChartCallback"), None)

    if progress_bar is not None and terminal_chart is not None:
        # We know these are the correct types based on the class name check above
        # avoiding circular imports for type checking here
        progress_bar.set_chart_callback(terminal_chart)  # type: ignore[attr-defined]

    trainer = WSTrainer(
        max_epochs=cfg.train.n_epochs,
        project_name="wsmart-route",
        experiment_name=cfg.experiment_name,
        accelerator=cfg.device if cfg.device != "cuda" else "auto",
        devices=cfg.train.devices,
        strategy=cfg.train.strategy,
        gradient_clip_val=(float(cfg.rl.max_grad_norm) if cfg.rl.algorithm != "ppo" else 0.0),
        logger=CSVLogger(cfg.train.logs_dir or "logs", name=""),
        callbacks=callbacks,  # type: ignore[arg-type]
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_step,
        model_weights_path=cfg.train.model_weights_path,
        logs_dir=cfg.train.logs_dir,
        reload_dataloaders_every_n_epochs=cfg.train.reload_dataloaders_every_n_epochs,
        enable_progress_bar=False,
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

    try:
        # Log training configuration as params
        _log_training_params(run, cfg)

        # Track validation dataset if available
        val_ds = cfg.train.val_dataset
        if val_ds and os.path.exists(str(val_ds)):
            run.log_dataset_event("load", file_path=str(val_ds))
            run.watch_file(str(val_ds))

        trainer.fit(model)

        # Save final weights if path is provided
        if cfg.train.final_model_path:
            model.save_weights(cfg.train.final_model_path)
            run.log_artifact(cfg.train.final_model_path, artifact_type="model")

        val_reward = trainer.callback_metrics.get("val/reward", torch.tensor(0.0)).item()
        run.log_metric("best/val_reward", val_reward)
        run.__exit__(None, None, None)
        return val_reward

    except Exception as exc:
        run.__exit__(type(exc), exc, exc.__traceback__)
        raise


def _log_training_params(run: wst.Run, cfg: Config) -> None:
    """Flatten and log relevant config sections as run parameters."""
    import omegaconf

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
