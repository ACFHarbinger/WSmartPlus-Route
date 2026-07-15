"""
HPO health metrics callback for Optuna / Ray Tune pruning (§A.4 Option D).

Reports gradient norm and policy entropy each validation epoch so HPO backends
can prune unstable trials early and WSTracker / WandB sweeps retain health data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore[assignment]

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment]


class HpoHealthMetricsCallback(Callback):
    """
    Log grad_norm and entropy for HPO trial pruning and sweep analytics.

    On each validation epoch:
    - Reports ``val/reward`` to an Optuna trial (when provided) for pruner hooks
    - Stores ``last_grad_norm`` / ``last_entropy`` user attributes on the trial
    - Logs ``hpo/grad_norm`` and ``hpo/entropy`` to the active WSTracker run
    - Optionally prunes trials that exceed health thresholds
    """

    def __init__(
        self,
        trial: Optional[Any] = None,
        max_grad_norm: float = 100.0,
        min_entropy: float = 0.01,
        prune_on_unhealthy: bool = True,
        reward_key: str = "val/reward",
        grad_norm_key: str = "train/grad_norm",
        entropy_key: str = "train/entropy",
        report_to_ray: bool = False,
    ) -> None:
        """
        Initialize health metric keys and pruning thresholds.

        Args:
            trial: Optional Optuna trial for intermediate reporting.
            max_grad_norm: Prune when gradient norm exceeds this value.
            min_entropy: Prune when entropy drops below this value.
            prune_on_unhealthy: Raise ``TrialPruned`` on threshold violations.
            reward_key: Lightning metric key for validation reward.
            grad_norm_key: Lightning metric key for gradient norm.
            entropy_key: Lightning metric key for policy entropy.
            report_to_ray: When True, emit per-epoch metrics via ``ray.train.report``.
        """
        super().__init__()
        self.trial = trial
        self.max_grad_norm = max_grad_norm
        self.min_entropy = min_entropy
        self.prune_on_unhealthy = prune_on_unhealthy
        self.reward_key = reward_key
        self.grad_norm_key = grad_norm_key
        self.entropy_key = entropy_key
        self.report_to_ray = report_to_ray

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Report health metrics and optionally prune unhealthy trials."""
        if not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        grad_norm = self._metric_value(metrics, self.grad_norm_key)
        entropy = self._metric_value(metrics, self.entropy_key)
        reward = self._metric_value(metrics, self.reward_key)

        self._log_to_tracker(grad_norm, entropy, reward, epoch)
        self._report_to_ray(grad_norm, entropy, reward, epoch)

        if self.trial is not None:
            self._report_to_optuna(grad_norm, entropy, reward, epoch)

    def _log_to_tracker(
        self,
        grad_norm: Optional[float],
        entropy: Optional[float],
        reward: Optional[float],
        epoch: int,
    ) -> None:
        if get_active_run is None:
            return
        run = get_active_run()
        if run is None:
            return
        if grad_norm is not None:
            run.log_metric("hpo/grad_norm", grad_norm, step=epoch)
        if entropy is not None:
            run.log_metric("hpo/entropy", entropy, step=epoch)
        if reward is not None:
            run.log_metric("hpo/val_reward", reward, step=epoch)

    def _report_to_ray(
        self,
        grad_norm: Optional[float],
        entropy: Optional[float],
        reward: Optional[float],
        epoch: int,
    ) -> None:
        if not self.report_to_ray:
            return
        try:
            import ray.train as ray_train
        except ImportError:
            return

        payload: Dict[str, Any] = {"training_iteration": epoch + 1}
        if reward is not None:
            payload["val_reward"] = reward
        if grad_norm is not None:
            payload["grad_norm"] = grad_norm
        if entropy is not None:
            payload["entropy"] = entropy
        if len(payload) > 1:
            ray_train.report(payload)

    def _report_to_optuna(
        self,
        grad_norm: Optional[float],
        entropy: Optional[float],
        reward: Optional[float],
        epoch: int,
    ) -> None:
        if optuna is None or self.trial is None:
            return

        if grad_norm is not None:
            self.trial.set_user_attr("last_grad_norm", grad_norm)
            self.trial.set_user_attr(f"grad_norm_epoch_{epoch}", grad_norm)
        if entropy is not None:
            self.trial.set_user_attr("last_entropy", entropy)
            self.trial.set_user_attr(f"entropy_epoch_{epoch}", entropy)

        if reward is not None:
            self.trial.report(reward, step=epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

        if not self.prune_on_unhealthy:
            return

        if grad_norm is not None and grad_norm > self.max_grad_norm:
            self.trial.set_user_attr("health_pruned", "grad_norm_explosion")
            raise optuna.TrialPruned(f"grad_norm={grad_norm:.4f}")

        if entropy is not None and entropy < self.min_entropy:
            self.trial.set_user_attr("health_pruned", "entropy_collapse")
            raise optuna.TrialPruned(f"entropy={entropy:.6f}")

    @staticmethod
    def _metric_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
        if key not in metrics:
            return None
        val = metrics[key]
        if hasattr(val, "item"):
            val = val.item()
        try:
            return float(val)
        except (TypeError, ValueError):
            return None


def apply_dehb_health_penalty(
    fitness: float,
    grad_norm: Optional[float],
    entropy: Optional[float],
    max_grad_norm: float = 100.0,
    min_entropy: float = 0.01,
) -> float:
    """Penalise DEHB fitness when trial health metrics are out of range."""
    penalised = fitness
    if grad_norm is not None and grad_norm > max_grad_norm:
        penalised += 1_000.0
    if entropy is not None and entropy < min_entropy:
        penalised += 500.0
    return penalised
