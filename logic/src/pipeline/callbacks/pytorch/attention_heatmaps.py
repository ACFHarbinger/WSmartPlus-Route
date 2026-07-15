"""
Attention heatmap logging callback for PyTorch Lightning (§A.2 Option C).

Captures runtime encoder attention during validation and logs heatmap images
to Weights & Biases / TensorBoard when tracking flags are enabled.
"""

from __future__ import annotations

from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger

from logic.src.tracking.logging.visualization.heatmaps import maybe_log_eval_attention_heatmaps


class AttentionHeatmapCallback(Callback):
    """Log attention heatmaps on validation epochs when configured.

    Respects ``tracking.log_attention`` (runtime matrices) and
    ``tracking.log_attention_heatmaps`` (Q/K/V weight plots).  Frequency is
    controlled by ``tracking.viz_every_n_epochs`` (``0`` = validation end only).
    """

    def __init__(self, tracking_cfg: Optional[Any] = None) -> None:
        """Store tracking configuration reference."""
        super().__init__()
        self._cfg = tracking_cfg

    def _enabled(self) -> bool:
        if self._cfg is None:
            return False
        return bool(
            getattr(self._cfg, "log_attention", False)
            or getattr(self._cfg, "log_attention_heatmaps", False)
        )

    def _should_run(self, epoch: int) -> bool:
        if not self._enabled():
            return False
        viz_n = int(getattr(self._cfg, "viz_every_n_epochs", 0) or 0)
        if viz_n <= 0:
            return True
        return epoch > 0 and epoch % viz_n == 0

    def _tb_writer(self, trainer: pl.Trainer) -> Any:
        for lg in trainer.loggers:
            if isinstance(lg, TensorBoardLogger):
                return lg.experiment
        return None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Capture attention from the first validation batch and log images."""
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        if not self._should_run(epoch):
            return

        try:
            val_loader = trainer.val_dataloaders
            if val_loader is None:
                return
            loader = val_loader[0] if isinstance(val_loader, list) else val_loader
            batch = next(iter(loader))
        except (StopIteration, TypeError, AttributeError):
            return

        class _CfgWrap:
            """Minimal config wrapper exposing ``tracking`` to eval helper."""

            def __init__(self, tracking: Any) -> None:
                self.tracking = tracking

        maybe_log_eval_attention_heatmaps(
            pl_module,
            batch,
            _CfgWrap(self._cfg),
            output_subdir="val_attention",
            step=trainer.global_step,
            tb_writer=self._tb_writer(trainer),
            phase="val",
            epoch=trainer.current_epoch,
        )
