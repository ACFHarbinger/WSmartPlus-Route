"""
Hyperparameter Optimization logic for training pipeline.

Backends
--------
* ``method in {"tpe", "random", "grid", "hyperband"}`` → :class:`OptunaHPO`
* ``method == "dehb"`` → :class:`DifferentialEvolutionHyperband`
* ``method in {"asha", "pbt", "bohb"}`` → :class:`RayTuneHPO`

MLflow dual-write
-----------------
When ``cfg.tracking.mlflow_enabled`` is ``True``, the HPO experiment is
wrapped in an MLflow parent run so aggregate results (best config, best
score, total wall-time) are visible in the MLflow UI alongside the
per-trial data logged by each backend.
"""

import contextlib
from typing import Any, Dict

import optuna
import torch
from omegaconf import OmegaConf

from logic.src.configs import Config
from logic.src.pipeline.features.train.model_factory import create_model
from logic.src.pipeline.rl.common.trainer import WSTrainer
from logic.src.pipeline.rl.hpo.base import apply_params, normalise_search_space
from logic.src.tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)

# Methods that delegate to Ray Tune
_RAY_TUNE_METHODS = {"asha", "pbt", "bohb"}


def objective(trial: optuna.Trial, base_cfg: Config) -> float:
    """Optuna objective function for HPO.

    Samples hyperparameters from the typed search space, applies them to
    a copy of the config, trains, and returns the metric to maximise.

    Args:
        trial: The Optuna trial object.
        base_cfg: The base configuration to copy and mutate.

    Returns:
        The validation reward (or ``-inf`` on failure).
    """
    from logic.src.pipeline.rl.hpo.base import BaseHPO

    try:
        from optuna.integration import PyTorchLightningPruningCallback
    except ImportError:
        from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

    # 1. Deep-copy config
    cfg = OmegaConf.to_object(base_cfg)
    assert isinstance(cfg, Config)

    # 2. Normalise and sample from the typed search space
    search_space = normalise_search_space(base_cfg.hpo.search_space)
    params: Dict[str, Any] = {}
    for name, spec in search_space.items():
        params[name] = BaseHPO.suggest_param_optuna(trial, name, spec)

    # 3. Apply sampled parameters to config
    apply_params(cfg, params)

    # 4. Build model and trainer
    model = create_model(cfg)

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val/reward")

    trainer = WSTrainer(
        max_epochs=cfg.hpo.n_epochs_per_trial,
        accelerator=cfg.device if cfg.device != "cuda" else "auto",
        devices=1 if cfg.device == "cuda" else "auto",
        enable_progress_bar=False,
        logger=False,
        callbacks=[pruning_callback],
        log_every_n_steps=cfg.tracking.log_step,
        tracking_cfg=cfg.tracking,
    )

    # 5. Train
    try:
        trainer.fit(model)
        return trainer.callback_metrics.get("val/reward", torch.tensor(float("-inf"))).item()
    except (KeyboardInterrupt, SystemExit):
        raise
    except optuna.exceptions.TrialPruned:
        raise
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        import traceback

        logger.error(f"Trial failed ({type(e).__name__}): {e}")
        logger.error(traceback.format_exc())
        return float("-inf")


def _ray_tune_objective(trial_cfg: Config) -> float:
    """Trainable passed to :class:`RayTuneHPO` — one epoch loop per trial.

    Trains for ``cfg.hpo.n_epochs_per_trial`` epochs and returns the
    validation reward.  Called inside a Ray worker process.

    Args:
        trial_cfg: Fully configured :class:`Config` with trial parameters
            already applied by :class:`RayTuneHPO`.

    Returns:
        Validation reward float (higher = better).
    """
    model = create_model(trial_cfg)
    trainer = WSTrainer(
        max_epochs=trial_cfg.hpo.n_epochs_per_trial,
        accelerator=trial_cfg.device if trial_cfg.device != "cuda" else "auto",
        devices=1 if trial_cfg.device == "cuda" else "auto",
        enable_progress_bar=False,
        logger=False,
        log_every_n_steps=trial_cfg.tracking.log_step,
        tracking_cfg=trial_cfg.tracking,
    )
    try:
        trainer.fit(model)
        return trainer.callback_metrics.get("val/reward", torch.tensor(float("-inf"))).item()
    except Exception:
        return float("-inf")


def run_hpo(cfg: Config) -> float:
    """Run Hyperparameter Optimization.

    Dispatches to the appropriate backend based on ``cfg.hpo.method``:

    * ``"dehb"`` → :class:`~logic.src.pipeline.rl.hpo.dehb.DifferentialEvolutionHyperband`
    * ``"asha"`` / ``"pbt"`` / ``"bohb"`` → :class:`~logic.src.pipeline.rl.hpo.ray_tune_hpo.RayTuneHPO`
    * anything else → :class:`~logic.src.pipeline.rl.hpo.optuna_hpo.OptunaHPO`

    When ``cfg.tracking.mlflow_enabled`` is ``True`` the entire HPO
    session is wrapped in an MLflow parent run so the best result and
    wall-time are surfaced in the MLflow UI.

    Args:
        cfg: Root application configuration.

    Returns:
        Best metric value found across all trials.
    """
    from logic.src.pipeline.rl.hpo import DifferentialEvolutionHyperband, OptunaHPO, RayTuneHPO

    # Enable Tensor Core acceleration for Ampere+ GPUs
    if torch.cuda.is_available() and cfg.train.precision in ["16-mixed", "bf16-mixed"]:
        torch.set_float32_matmul_precision("medium")

    # Normalise the search space once (shared by all backends)
    search_space = normalise_search_space(cfg.hpo.search_space)

    # Resolve tracking config fields safely (defaults if attribute absent)
    tracking = getattr(cfg, "tracking", None)

    # ----- ZenML dispatch (opt-in) -----
    zenml_enabled = bool(getattr(tracking, "zenml_enabled", False))
    if zenml_enabled:
        return _run_hpo_via_zenml(cfg)

    mlflow_enabled = bool(getattr(tracking, "mlflow_enabled", False))
    mlflow_uri = str(getattr(tracking, "mlflow_tracking_uri", "mlruns"))
    mlflow_exp = str(getattr(tracking, "mlflow_experiment_name", "wsmart-route"))
    ray_mlflow = bool(getattr(tracking, "ray_tune_mlflow_enabled", False))

    best_val = 0.0

    with _mlflow_hpo_run(mlflow_enabled, mlflow_uri, mlflow_exp, cfg) as _mlflow_run:
        # ------------------------------------------------------------------
        # 1. DEHB
        # ------------------------------------------------------------------
        if cfg.hpo.method == "dehb":

            def dehb_obj(config: Dict[str, Any], fidelity: float) -> Dict[str, float]:
                temp_cfg = OmegaConf.to_object(cfg)
                assert isinstance(temp_cfg, Config)
                config_dict = config.get_dictionary() if hasattr(config, "get_dictionary") else dict(config)
                apply_params(temp_cfg, config_dict)
                model = create_model(temp_cfg)
                trainer = WSTrainer(
                    max_epochs=int(fidelity),
                    enable_progress_bar=False,
                    logger=False,
                    log_every_n_steps=temp_cfg.tracking.log_step,
                    tracking_cfg=temp_cfg.tracking,
                )
                trainer.fit(model)
                reward = trainer.callback_metrics.get("val/reward", torch.tensor(0.0)).item()
                return {"fitness": -reward}

            dehb = DifferentialEvolutionHyperband(
                cfg=cfg,
                objective_fn=dehb_obj,
                search_space=search_space,
                min_fidelity=getattr(cfg.hpo, "min_fidelity", 1) or 1,
                max_fidelity=cfg.hpo.n_epochs_per_trial,
            )
            best_val = dehb.run()
            logger.info(f"DEHB complete in {dehb.runtime:.2f}s. Best config: {dehb.best_config}")

            # Forward best result to MLflow parent run
            with contextlib.suppress(Exception):
                if _mlflow_run is not None:
                    import mlflow  # type: ignore[import-not-found]

                    mlflow.log_metric("hpo/best_val_reward", best_val)
                    mlflow.log_params({str(k): str(v) for k, v in (dehb.best_config or {}).items()})

        # ------------------------------------------------------------------
        # 2. Ray Tune (ASHA / PBT / BOHB)
        # ------------------------------------------------------------------
        elif cfg.hpo.method in _RAY_TUNE_METHODS:
            ray_mlflow_uri = mlflow_uri if (mlflow_enabled and ray_mlflow) else None
            ray_mlflow_exp = mlflow_exp if ray_mlflow_uri else None

            hpo_runner: Any = RayTuneHPO(
                cfg=cfg,
                objective_fn=_ray_tune_objective,
                search_space=search_space,
                scheduler=cfg.hpo.method,
                mlflow_tracking_uri=ray_mlflow_uri,
                mlflow_experiment_name=ray_mlflow_exp,
            )
            best_val = hpo_runner.run()
            logger.info(f"Ray Tune ({cfg.hpo.method}) complete. Best val_reward: {best_val:.4f}")

            with contextlib.suppress(Exception):
                if _mlflow_run is not None:
                    import mlflow  # type: ignore[import-not-found]

                    mlflow.log_metric("hpo/best_val_reward", best_val)

        # ------------------------------------------------------------------
        # 3. Optuna (TPE / Grid / Random / Hyperband)
        # ------------------------------------------------------------------
        else:
            hpo_runner = OptunaHPO(cfg, objective, search_space=search_space)
            best_val = hpo_runner.run()
            logger.info(f"Optuna ({cfg.hpo.method}) complete. Best val_reward: {best_val:.4f}")

            with contextlib.suppress(Exception):
                if _mlflow_run is not None:
                    import mlflow  # type: ignore[import-not-found]

                    mlflow.log_metric("hpo/best_val_reward", best_val)

    return best_val


# ---------------------------------------------------------------------------
# MLflow context manager for the HPO session
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _mlflow_hpo_run(
    enabled: bool,
    tracking_uri: str,
    experiment_name: str,
    cfg: Config,
):
    """Context manager that wraps the HPO session in an MLflow run.

    When *enabled* is ``False`` it yields ``None`` without touching MLflow.
    All exceptions from MLflow setup are suppressed so a broken server cannot
    block the HPO run.  Exceptions raised inside the ``with`` body are always
    propagated — only the *setup* phase is guarded.

    Yields:
        The active ``mlflow.ActiveRun`` context, or ``None``.
    """
    if not enabled:
        yield None
        return

    # ------------------------------------------------------------------ #
    # Setup phase — safe to catch and fall back to None                   #
    # ------------------------------------------------------------------ #
    mlflow_run = None
    try:
        import mlflow  # type: ignore[import-not-found]

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        hpo_tags = {
            "task": "hpo",
            "method": cfg.hpo.method,
            "n_trials": str(cfg.hpo.n_trials or cfg.hpo.num_samples),
            "problem": str(getattr(cfg.env, "name", "")),
        }
        mlflow_run = mlflow.start_run(run_name=f"hpo-{cfg.hpo.method}", tags=hpo_tags)
        active_run = mlflow_run.__enter__()
        with contextlib.suppress(Exception):
            mlflow.log_params(
                {
                    "hpo.method": cfg.hpo.method,
                    "hpo.n_epochs_per_trial": cfg.hpo.n_epochs_per_trial,
                    "hpo.num_samples": cfg.hpo.num_samples,
                    "env.name": str(getattr(cfg.env, "name", "")),
                    "env.num_loc": str(getattr(cfg.env, "num_loc", "")),
                }
            )
    except Exception as exc:
        logger.warning(f"MLflow HPO run setup failed (continuing without MLflow): {exc}")
        yield None
        return

    # ------------------------------------------------------------------ #
    # Yield phase — mlflow_run is live; always clean up in finally        #
    # ------------------------------------------------------------------ #
    try:
        yield active_run
    finally:
        with contextlib.suppress(Exception):
            mlflow_run.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ZenML dispatch
# ---------------------------------------------------------------------------


def _run_hpo_via_zenml(cfg: Config) -> float:
    """Dispatch HPO to the ZenML HPO pipeline.

    Called when ``cfg.tracking.zenml_enabled`` is ``True``.

    Returns:
        Best metric value found, or ``0.0`` on ZenML failure.
    """
    tracking = getattr(cfg, "tracking", None)
    mlflow_uri = str(getattr(tracking, "mlflow_tracking_uri", "mlruns"))
    stack_name = str(getattr(tracking, "zenml_stack_name", "wsmart-route-stack"))

    from logic.src.tracking.integrations.zenml_bridge import configure_zenml_stack

    if not configure_zenml_stack(mlflow_uri, stack_name=stack_name):
        logger.warning("ZenML stack configuration failed — falling back to direct HPO.")
        # Disable ZenML to avoid infinite recursion
        if tracking is not None:
            tracking.zenml_enabled = False  # type: ignore[union-attr]
        return run_hpo(cfg)

    try:
        from logic.src.pipeline.rl.hpo.zenml_hpo_pipeline import hpo_pipeline

        result = hpo_pipeline(cfg)
        return result if isinstance(result, float) else 0.0
    except Exception as exc:
        logger.warning(f"ZenML HPO pipeline failed — falling back to direct HPO: {exc}")
        if tracking is not None:
            tracking.zenml_enabled = False  # type: ignore[union-attr]
        return run_hpo(cfg)
