"""PyTorch Lightning callback that forwards all training events to WSTracker."""

from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from logic.src.tracking.core.run import get_active_run


class TrackingCallback(Callback):
    """Lightning callback that logs training metrics and artifacts to WSTracker.

    Hooks into the Lightning trainer lifecycle to:

    * Log model hyperparameters at training start.
    * Log ``train/*`` metrics at the end of every training epoch.
    * Log ``val/*`` metrics at the end of every validation epoch.
    * Log ``time/*`` speed-monitor metrics at training end.
    * Register the best and final model checkpoints as run artifacts.
    * Record dataset regeneration events when dataloaders are reloaded.

    This callback is added automatically by :class:`WSTrainer` and requires
    no user configuration.
    """

    # ------------------------------------------------------------------
    # Fit lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        run = get_active_run()
        if run is None:
            return

        # --- Hyperparameters ---
        hparams: Dict[str, Any] = {}
        if hasattr(pl_module, "hparams"):
            with contextlib.suppress(Exception):
                hparams.update(dict(pl_module.hparams))

        hparams.update(
            {
                "trainer.max_epochs": trainer.max_epochs,
                "trainer.precision": str(trainer.precision),
                "trainer.gradient_clip_val": trainer.gradient_clip_val,
                "trainer.log_every_n_steps": trainer.log_every_n_steps,
            }
        )
        run.log_params(hparams)
        run.set_tag("status", "training")

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        run = get_active_run()
        if run is None:
            return
        epoch = trainer.current_epoch
        metrics = _extract_metrics(trainer.callback_metrics, prefix="train/")
        run.log_metrics(metrics, step=epoch)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        run = get_active_run()
        if run is None:
            return
        epoch = trainer.current_epoch
        metrics = _extract_metrics(trainer.callback_metrics, prefix="val/")
        run.log_metrics(metrics, step=epoch)

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        run = get_active_run()
        if run is None:
            return

        # Log SpeedMonitor timing metrics
        time_metrics = _extract_metrics(trainer.callback_metrics, prefix="time/")
        if time_metrics:
            run.log_metrics(time_metrics, step=trainer.current_epoch)

        # Register best and last checkpoints
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                _log_checkpoint_artifact(run, cb, trainer.current_epoch)

        run.set_tag("status", "completed")
        run.set_tag("final_epoch", str(trainer.current_epoch))
        run.flush()

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log gradient L2 norm and current learning rate."""
        run = get_active_run()
        if run is None:
            return
        step = trainer.global_step

        # Gradient L2 norm
        total_norm_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        run.log_metric("train/grad_norm", total_norm_sq**0.5, step=step)

        # Learning rate (first param group)
        for i, pg in enumerate(optimizer.param_groups):
            key = "train/lr" if i == 0 else f"train/lr_group{i}"
            run.log_metric(key, pg["lr"], step=step)

    # ------------------------------------------------------------------
    # Checkpoint events (fires after each ModelCheckpoint save)
    # ------------------------------------------------------------------

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        run = get_active_run()
        if run is None:
            return
        val_reward = trainer.callback_metrics.get("val/reward")
        epoch = trainer.current_epoch
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                path = cb.best_model_path
                if os.path.exists(path):
                    run.log_artifact(
                        path,
                        artifact_type="checkpoint",
                        metadata={
                            "epoch": epoch,
                            "val_reward": float(val_reward) if val_reward is not None else None,
                        },
                    )

    # ------------------------------------------------------------------
    # Dataset regeneration detection
    # ------------------------------------------------------------------

    def on_train_dataloader(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Record a dataset mutation event when the dataloader is (re)loaded."""
        run = get_active_run()
        if run is None:
            return
        epoch = trainer.current_epoch
        if epoch == 0:
            return  # Skip initial load; it's already covered by log_dataset_event in the engine
        run.log_dataset_event(
            "mutate",
            metadata={
                "description": "Epoch dataloader reload",
                "epoch": epoch,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_metrics(
    callback_metrics: Dict[str, Any],
    prefix: str,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for k, v in callback_metrics.items():
        if k.startswith(prefix):
            with contextlib.suppress(TypeError, ValueError):
                result[k] = float(v.item() if hasattr(v, "item") else v)
    return result


def _log_checkpoint_artifact(
    run: Any,
    cb: ModelCheckpoint,
    epoch: int,
) -> None:
    for attr, label in [("best_model_path", "best_checkpoint"), ("last_model_path", "last_checkpoint")]:
        path: Optional[str] = getattr(cb, attr, None)
        if path and os.path.exists(path):
            run.log_artifact(
                path,
                name=label,
                artifact_type="checkpoint",
                metadata={
                    "epoch": epoch,
                    "monitor": cb.monitor,
                    "best": attr == "best_model_path",
                },
            )
