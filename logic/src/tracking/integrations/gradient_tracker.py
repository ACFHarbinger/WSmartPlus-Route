"""Gradient and weight distribution tracker for PyTorch Lightning.

This module provides the :class:`GradientTrackerCallback` which monitors
gradient norms, weight distributions, and stability ratios during training.
It integrates with W&B and TensorBoard to provide visual diagnostics for neural
routing model convergence.

Attributes:
    GradientTrackerCallback: Callback for monitoring gradient and weight dynamics.

Example:
    >>> from logic.src.tracking.integrations.gradient_tracker import GradientTrackerCallback
    >>> tracker = GradientTrackerCallback(log_freq=10, prefix="train/debug/")
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, cast

from pytorch_lightning.callbacks import Callback

if TYPE_CHECKING:
    import pytorch_lightning as pl
    import torch.nn as nn

try:
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
except ImportError:
    TensorBoardLogger = None  # type: ignore[assignment,misc]
    WandbLogger = None  # type: ignore[assignment,misc]

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


class GradientTrackerCallback(Callback):
    """Tracks gradient norms and weight distributions per layer during training.

    Logs diagnostic metrics including global and per-layer gradient norms,
    grad/weight ratios, and optional histograms to the active logger.

    Attributes:
        log_freq: Step frequency for scalar logging.
        hist_freq: Step frequency for histogram logging.
        norm_type: L-norm type for gradient calculation.
        log_histograms: If True, enables weight/gradient histogram capture.
        include_bias: If True, tracks bias parameters in addition to weights.
        layer_filter: List of substrings to filter tracked layer names.
        prefix: Namespace prefix for all logged metric keys.
        max_grad_norm_alert: Threshold for emitting gradient explosion warnings.
    """

    def __init__(
        self,
        log_freq: int = 50,
        hist_freq: int = 200,
        norm_type: float = 2.0,
        log_histograms: bool = True,
        include_bias: bool = False,
        layer_filter: Optional[List[str]] = None,
        prefix: str = "debug/",
        max_grad_norm_alert: Optional[float] = 100.0,
    ) -> None:
        """Initializes the gradient tracking callback.

        Args:
            log_freq: Scalar logging frequency. Defaults to 50.
            hist_freq: Histogram logging frequency. Defaults to 200.
            norm_type: Order of the norm (1, 2, or inf). Defaults to 2.0.
            log_histograms: Whether to log distribution histograms. Defaults to True.
            include_bias: Whether to track bias terms. Defaults to False.
            layer_filter: Optional substrings to filter parameter names.
            prefix: Metric key prefix. Defaults to "debug/".
            max_grad_norm_alert: Gradient norm threshold for runtime warnings.
        """
        super().__init__()
        self.log_freq = log_freq
        self.hist_freq = hist_freq
        self.norm_type = norm_type
        self.log_histograms = log_histograms
        self.include_bias = include_bias
        self.layer_filter: Optional[List[str]] = layer_filter
        self.prefix = prefix
        self.max_grad_norm_alert = max_grad_norm_alert

        self._step: int = 0
        self._alerted_steps: Set[int] = set()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_track(self, name: str) -> bool:
        """Determines if a parameter should be included in tracking.

        Args:
            name: Parameter name (e.g., 'encoder.layers.0.weight').

        Returns:
            bool: True if the parameter passes all filters.
        """
        if not self.include_bias and name.endswith(".bias"):
            return False
        if self.layer_filter is None:
            return True
        return any(substr in name for substr in self.layer_filter)

    def _grad_norm(self, param: nn.Parameter) -> Optional[float]:
        """Calculates the gradient norm for a single parameter tensor.

        Args:
            param: The parameter object to inspect.

        Returns:
            Optional[float]: The computed norm, or None if no gradient exists.
        """
        if param.grad is None:
            return None
        if math.isinf(self.norm_type):
            return param.grad.data.abs().max().item()
        return param.grad.data.norm(self.norm_type).item()

    def _detect_logger(self, trainer: pl.Trainer) -> Optional[str]:
        """Identifies the active experiment logger type.

        Args:
            trainer: The PyTorch Lightning trainer instance.

        Returns:
            Optional[str]: 'wandb', 'tensorboard', or None.
        """
        if WandbLogger is None or TensorBoardLogger is None:
            return None
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return "wandb"
            if isinstance(logger, TensorBoardLogger):
                return "tensorboard"
        return None

    # ------------------------------------------------------------------
    # Lightning lifecycle hooks
    # ------------------------------------------------------------------

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Hook called immediately after backward pass to log gradient norms.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Active Lightning module.
        """
        if not trainer.is_global_zero:
            return

        self._step = trainer.global_step
        if self._step % self.log_freq != 0:
            return

        log_dict: Dict[str, float] = {}
        global_norm_sq = 0.0
        n_tracked = 0

        for name, param in pl_module.named_parameters():
            if not self._should_track(name):
                continue
            g_norm = self._grad_norm(param)
            if g_norm is None:
                continue

            n_tracked += 1
            global_norm_sq += g_norm**2
            safe = name.replace(".", "/")
            log_dict[f"{self.prefix}grad_norm/{safe}"] = g_norm

            # Grad/weight ratio — stability diagnostic
            w_norm = param.data.norm(self.norm_type).item()
            if w_norm > 1e-8:
                log_dict[f"{self.prefix}grad_weight_ratio/{safe}"] = g_norm / w_norm

        if n_tracked == 0:
            return

        global_norm = math.sqrt(global_norm_sq)
        log_dict[f"{self.prefix}global_grad_norm"] = global_norm

        pl_module.log_dict(log_dict, on_step=True, on_epoch=False, prog_bar=False)

        # Emit warning on gradient explosion (once per step)
        if (
            self.max_grad_norm_alert is not None
            and global_norm > self.max_grad_norm_alert
            and self._step not in self._alerted_steps
        ):
            self._alerted_steps.add(self._step)
            warnings.warn(
                f"[GradientTracker] Step {self._step}: global grad norm "
                f"{global_norm:.2f} exceeds threshold {self.max_grad_norm_alert}. "
                "Consider adding gradient clipping (gradient_clip_val in Trainer).",
                RuntimeWarning,
                stacklevel=2,
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Hook called at batch end to log distribution histograms.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Active Lightning module.
            outputs: Batch outputs.
            batch: Data batch.
            batch_idx: Global batch index.
        """
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        if not self.log_histograms or step % self.hist_freq != 0:
            return

        logger_type = self._detect_logger(trainer)
        if logger_type == "wandb":
            self._log_histograms_wandb(pl_module, step)
        elif logger_type == "tensorboard":
            self._log_histograms_tensorboard(trainer, pl_module, step)

    # ------------------------------------------------------------------
    # Logger-specific histogram methods
    # ------------------------------------------------------------------

    def _log_histograms_wandb(
        self,
        pl_module: pl.LightningModule,
        step: int,
    ) -> None:
        """Serializes and sends histograms to Weights & Biases.

        Args:
            pl_module: Lightning module containing parameters.
            step: Global step index.
        """
        if wandb is None or wandb.run is None:
            return

        hist_data: Dict[str, Any] = {}
        for name, param in pl_module.named_parameters():
            if not self._should_track(name):
                continue
            safe = name.replace(".", "/")
            hist_data[f"{self.prefix}weights/{safe}"] = wandb.Histogram(
                cast(Any, param.data.detach().cpu().float().numpy())
            )
            if param.grad is not None:
                hist_data[f"{self.prefix}gradients/{safe}"] = wandb.Histogram(
                    cast(Any, param.grad.detach().cpu().float().numpy())
                )

        wandb.log(hist_data, step=step)

    def _log_histograms_tensorboard(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        step: int,
    ) -> None:
        """Serializes and sends histograms to TensorBoard.

        Args:
            trainer: Lightning trainer instance.
            pl_module: Lightning module containing parameters.
            step: Global step index.
        """
        if TensorBoardLogger is None:
            return

        for logger in trainer.loggers:
            if not isinstance(logger, TensorBoardLogger):
                continue
            writer = logger.experiment
            for name, param in pl_module.named_parameters():
                if not self._should_track(name):
                    continue
                safe = name.replace(".", "/")
                writer.add_histogram(
                    f"{self.prefix}weights/{safe}",
                    param.data.detach().cpu(),
                    global_step=step,
                )
                if param.grad is not None:
                    writer.add_histogram(
                        f"{self.prefix}gradients/{safe}",
                        param.grad.detach().cpu(),
                        global_step=step,
                    )
            break  # Only log to the first TensorBoard logger found
