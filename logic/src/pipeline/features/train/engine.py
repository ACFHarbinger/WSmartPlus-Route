"""
Training engine for WSmart-Route.
"""

import contextlib

import hydra
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger

from logic.src.configs import Config
from logic.src.interfaces import ITraversable
from logic.src.pipeline.callbacks import SpeedMonitor
from logic.src.pipeline.features.train.model_factory.builder import create_model
from logic.src.pipeline.rl.common.trainer import WSTrainer


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

    # DEBUG: Print callbacks
    print(f"[Engine] Active callbacks: {[type(c).__name__ for c in callbacks]}")

    # Link Progress Bar and Chart if both exist
    # This is necessary because they are instantiated separately by Hydra
    progress_bar = next((c for c in callbacks if c.__class__.__name__ == "CleanProgressBar"), None)
    terminal_chart = next((c for c in callbacks if c.__class__.__name__ == "TerminalChartCallback"), None)

    if progress_bar is not None and terminal_chart is not None:
        # We know these are the correct types based on the class name check above
        # avoiding circular imports for type checking here
        progress_bar.set_chart_callback(terminal_chart)

    trainer = WSTrainer(
        max_epochs=cfg.train.n_epochs,
        project_name="wsmart-route",
        experiment_name=cfg.experiment_name,
        accelerator=cfg.device if cfg.device != "cuda" else "auto",
        devices=cfg.train.devices,
        strategy=cfg.train.strategy,
        gradient_clip_val=(float(cfg.rl.max_grad_norm) if cfg.rl.algorithm != "ppo" else 0.0),
        logger=CSVLogger(cfg.train.logs_dir or "logs", name=""),
        callbacks=callbacks,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_step,
        model_weights_path=cfg.train.model_weights_path,
        logs_dir=cfg.train.logs_dir,
        reload_dataloaders_every_n_epochs=cfg.train.reload_dataloaders_every_n_epochs,
        enable_progress_bar=False,
    )

    trainer.fit(model)

    # Save final weights if path is provided
    if cfg.train.final_model_path:
        model.save_weights(cfg.train.final_model_path)

    return trainer.callback_metrics.get("val/reward", torch.tensor(0.0)).item()
