"""
Custom PyTorch Lightning Trainer for WSmart-Route.
"""
from __future__ import annotations

from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import Logger, WandbLogger


class WSTrainer(pl.Trainer):
    """
    Extended Trainer with WSmart-Route specific features.

    Provides sensible defaults for:
    - Checkpointing
    - Progress bars
    - Logging
    """

    def __init__(
        self,
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: Union[int, str] = "auto",
        gradient_clip_val: float = 1.0,
        log_every_n_steps: int = 50,
        check_val_every_n_epoch: int = 1,
        project_name: str = "wsmart-route",
        experiment_name: Optional[str] = None,
        callbacks: Optional[list[Callback]] = None,
        logger: Optional[Union[Logger, bool]] = None,
        enable_progress_bar: bool = True,
        **kwargs,
    ):
        # Build callbacks
        callbacks = callbacks or []
        callbacks = self._add_default_callbacks(callbacks, enable_progress_bar)

        # Build logger
        if logger is None:
            logger = self._create_default_logger(project_name, experiment_name)

        super().__init__(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            gradient_clip_val=gradient_clip_val,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=check_val_every_n_epoch,
            callbacks=callbacks,
            logger=logger,
            **kwargs,
        )

    def _add_default_callbacks(
        self,
        callbacks: list[Callback],
        enable_progress_bar: bool,
    ) -> list[Callback]:
        """Add default callbacks if not present."""
        callback_types = {type(c) for c in callbacks}

        # Add checkpoint callback
        if ModelCheckpoint not in callback_types:
            callbacks.append(
                ModelCheckpoint(
                    monitor="val/reward",
                    mode="max",
                    save_top_k=3,
                    filename="{epoch}-{val_reward:.4f}",
                    save_last=True,
                )
            )

        # Add progress bar
        if enable_progress_bar and RichProgressBar not in callback_types:
            callbacks.append(RichProgressBar())

        return callbacks

    def _create_default_logger(
        self,
        project_name: str,
        experiment_name: Optional[str],
    ) -> Logger:
        """Create default WandB logger."""
        try:
            return WandbLogger(
                project=project_name,
                name=experiment_name,
                log_model=False,
            )
        except Exception:
            # Fall back to TensorBoard if WandB not available
            from pytorch_lightning.loggers import TensorBoardLogger

            return TensorBoardLogger("lightning_logs", name=experiment_name)
