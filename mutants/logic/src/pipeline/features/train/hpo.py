"""
Hyperparameter Optimization logic for training pipeline.
"""

from typing import Dict, Tuple, Union, cast

import optuna
import torch
from logic.src.configs import Config
from logic.src.pipeline.features.train.model_factory import create_model
from logic.src.pipeline.rl.common.trainer import WSTrainer
from logic.src.utils.logging.pylogger import get_pylogger
from omegaconf import DictConfig, OmegaConf

logger = get_pylogger(__name__)


def objective(trial: optuna.Trial, base_cfg: Config) -> float:
    """Optuna objective function for HPO."""
    from optuna.integration import PyTorchLightningPruningCallback

    # 1. Sample Hyperparameters
    cfg = OmegaConf.to_object(base_cfg)
    assert isinstance(cfg, Config)

    # Map search space from config to trial suggestions
    for key, range_val in base_cfg.hpo.search_space.items():
        if isinstance(range_val[0], float):
            val = trial.suggest_float(
                key,
                range_val[0],
                range_val[1],
                log=(True if range_val[0] > 0 and range_val[1] / range_val[0] > 10 else False),
            )
        elif isinstance(range_val[0], int):
            val = trial.suggest_int(key, range_val[0], range_val[1])
        else:
            val = trial.suggest_categorical(key, range_val)

        # Recursive attribute setting (e.g., 'optim.lr')
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)

    # 2. Initialize Model and Trainer
    model = create_model(cfg)

    # Use pruning callback
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

    # 3. Train
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
    """Run Hyperparameter Optimization."""
    from logic.src.pipeline.rl.hpo import DifferentialEvolutionHyperband, OptunaHPO

    # Enable Tensor Core acceleration for Ampere+ GPUs
    if torch.cuda.is_available() and cfg.train.precision in ["16-mixed", "bf16-mixed"]:
        torch.set_float32_matmul_precision("medium")

    # 1. DEHB Method
    if cfg.hpo.method == "dehb":

        def dehb_obj(config, fidelity):
            """DEHB objective function."""
            temp_cfg = OmegaConf.to_object(cfg)
            # Update config with suggested values
            for k, v in config.items():
                parts = k.split(".")
                obj = temp_cfg
                for part in parts[:-1]:
                    if obj is not None:
                        obj = getattr(obj, part)
                setattr(obj, parts[-1], v)

            model = create_model(cast(DictConfig, temp_cfg))
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
            cs=cast(Dict[str, Union[Tuple[float, float], list]], cfg.hpo.search_space),
            f=dehb_obj,
            min_fidelity=getattr(cfg.hpo, "min_epochs", 1) or 1,
            max_fidelity=cfg.hpo.n_epochs_per_trial,
        )
        best_config, runtime, _ = dehb.run(fevals=cfg.hpo.n_trials)
        logger.info(f"DEHB complete in {runtime:.2f}s. Best config: {best_config}")
        return 0.0

    # 2. Optuna Methods (TPE, Grid, Random, Hyperband)
    hpo_runner = OptunaHPO(cfg, objective)
    return hpo_runner.run()
