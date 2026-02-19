"""
Hyperparameter Optimization logic for training pipeline.
"""

from typing import Any, Dict

import optuna
import torch
from omegaconf import OmegaConf

from logic.src.configs import Config
from logic.src.pipeline.features.train.model_factory import create_model
from logic.src.pipeline.rl.common.trainer import WSTrainer
from logic.src.pipeline.rl.hpo.base import apply_params, normalise_search_space
from logic.src.utils.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)


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
        log_every_n_steps=cfg.train.log_step,
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


def run_hpo(cfg: Config) -> float:
    """Run Hyperparameter Optimization.

    Chooses the backend (DEHB or Optuna) based on ``cfg.hpo.method`` and
    executes the search over the typed search space defined in
    ``cfg.hpo.search_space``.

    Args:
        cfg: Root application configuration.

    Returns:
        The best metric value found.
    """
    from logic.src.pipeline.rl.hpo import DifferentialEvolutionHyperband, OptunaHPO

    # Enable Tensor Core acceleration for Ampere+ GPUs
    if torch.cuda.is_available() and cfg.train.precision in ["16-mixed", "bf16-mixed"]:
        torch.set_float32_matmul_precision("medium")

    # Normalise the search space once
    search_space = normalise_search_space(cfg.hpo.search_space)

    # 1. DEHB Method
    if cfg.hpo.method == "dehb":

        def dehb_obj(config: Dict[str, Any], fidelity: float) -> Dict[str, float]:
            """DEHB objective function (minimises fitness)."""
            temp_cfg = OmegaConf.to_object(cfg)
            assert isinstance(temp_cfg, Config)

            # Convert ConfigSpace config to plain dict if needed
            config_dict = config.get_dictionary() if hasattr(config, "get_dictionary") else dict(config)
            apply_params(temp_cfg, config_dict)

            model = create_model(temp_cfg)
            trainer = WSTrainer(
                max_epochs=int(fidelity),
                enable_progress_bar=False,
                logger=False,
                log_every_n_steps=temp_cfg.train.log_step,
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
        return best_val

    # 2. Optuna Methods (TPE, Grid, Random, Hyperband)
    hpo_runner = OptunaHPO(cfg, objective, search_space=search_space)
    return hpo_runner.run()
