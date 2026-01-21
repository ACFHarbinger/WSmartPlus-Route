"""
Unified Training and Hyperparameter Optimization entry point using PyTorch Lightning and Hydra.
"""


import hydra
import optuna
import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from logic.src.callbacks import SpeedMonitor
from logic.src.configs import Config
from logic.src.envs import get_env
from logic.src.models.policies import (
    AttentionModelPolicy,
    DeepDecoderPolicy,
    PointerNetworkPolicy,
    SymNCOPolicy,
    TemporalAMPolicy,
)
from logic.src.models.policies.classical.alns import ALNSPolicy
from logic.src.models.policies.classical.hgs import HGSPolicy
from logic.src.models.policies.classical.hybrid import NeuralHeuristicHybrid
from logic.src.pipeline.rl import (
    DRGRPO,
    GSPO,
    POMO,
    PPO,
    REINFORCE,
    SAPO,
    HRLModule,
    MetaRLModule,
    SymNCO,
)
from logic.src.pipeline.trainer import WSTrainer
from logic.src.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)

# Register configuration
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def create_model(cfg: Config) -> pl.LightningModule:
    """Helper to create the RL model based on config."""
    # 1. Initialize Environment
    # If cfg.env is a dataclass, we can access attributes directly
    env_name = cfg.env.name
    env_kwargs = {k: v for k, v in vars(cfg.env).items() if k != "name"}
    env = get_env(env_name, **env_kwargs)

    # 2. Initialize Policy
    policy_map = {
        "am": AttentionModelPolicy,
        "deep_decoder": DeepDecoderPolicy,
        "temporal": TemporalAMPolicy,
        "pointer": PointerNetworkPolicy,
        "symnco": SymNCOPolicy,
        "alns": ALNSPolicy,
        "hgs": HGSPolicy,
        "hybrid": NeuralHeuristicHybrid,
    }

    if cfg.model.name not in policy_map:
        raise ValueError(f"Unknown model name: {cfg.model.name}. Available: {list(policy_map.keys())}")

    policy_cls = policy_map[cfg.model.name]

    policy_kwargs = {
        "env_name": cfg.env.name,
        "embed_dim": cfg.model.embed_dim,
        "hidden_dim": cfg.model.hidden_dim,
        "n_encode_layers": cfg.model.num_encoder_layers,
        "n_heads": cfg.model.num_heads,
        "normalization": "batch",
    }
    if cfg.model.name == "deep_decoder":
        policy_kwargs["n_decode_layers"] = cfg.model.num_decoder_layers

    if cfg.model.name == "hybrid":
        neural = AttentionModelPolicy(**policy_kwargs)
        heuristic = ALNSPolicy(env_name=cfg.env.name)
        policy = NeuralHeuristicHybrid(neural, heuristic)
    else:
        policy = policy_cls(**policy_kwargs)

    # 3. Initialize RL Module
    common_kwargs = {
        "env": env,
        "policy": policy,
        "optimizer": cfg.optim.optimizer,
        "optimizer_kwargs": {"lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay},
        "lr_scheduler": cfg.optim.lr_scheduler,
        "lr_scheduler_kwargs": cfg.optim.lr_scheduler_kwargs,
        "train_data_size": cfg.train.train_data_size,
        "val_data_size": cfg.train.val_data_size,
        "val_dataset_path": cfg.train.val_dataset,
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "entropy_weight": cfg.rl.entropy_weight,
        "max_grad_norm": cfg.rl.max_grad_norm,
    }

    if cfg.rl.algorithm == "ppo":
        from logic.src.models.policies.critic import create_critic_from_actor

        critic = create_critic_from_actor(
            policy,
            env_name=cfg.env.name,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.num_encoder_layers,
            n_heads=cfg.model.num_heads,
        )
        model: pl.LightningModule = PPO(
            critic=critic,
            ppo_epochs=cfg.rl.ppo_epochs,
            eps_clip=cfg.rl.eps_clip,
            value_loss_weight=cfg.rl.value_loss_weight,
            **common_kwargs,
        )
    elif cfg.rl.algorithm == "sapo":
        from logic.src.models.policies.critic import create_critic_from_actor

        critic = create_critic_from_actor(
            policy,
            env_name=cfg.env.name,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.num_encoder_layers,
            n_heads=cfg.model.num_heads,
        )
        model = SAPO(
            critic=critic,
            tau_pos=cfg.rl.sapo_tau_pos,
            tau_neg=cfg.rl.sapo_tau_neg,
            ppo_epochs=cfg.rl.ppo_epochs,
            **common_kwargs,
        )
    elif cfg.rl.algorithm == "gspo":
        from logic.src.models.policies.critic import create_critic_from_actor

        critic = create_critic_from_actor(
            policy,
            env_name=cfg.env.name,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.num_encoder_layers,
            n_heads=cfg.model.num_heads,
        )
        model = GSPO(
            critic=critic,
            ppo_epochs=cfg.rl.ppo_epochs,
            **common_kwargs,
        )
    elif cfg.rl.algorithm == "dr_grpo":
        from logic.src.models.policies.critic import create_critic_from_actor

        critic = create_critic_from_actor(
            policy,
            env_name=cfg.env.name,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.num_encoder_layers,
            n_heads=cfg.model.num_heads,
        )
        model = DRGRPO(
            critic=critic,
            ppo_epochs=cfg.rl.ppo_epochs,
            **common_kwargs,
        )
    elif cfg.rl.algorithm == "pomo":
        model = POMO(
            num_augment=cfg.rl.num_augment,
            augment_fn=cfg.rl.augment_fn,
            num_starts=cfg.rl.num_starts,
            **common_kwargs,
        )
    elif cfg.rl.algorithm == "symnco":
        model = SymNCO(
            alpha=cfg.rl.symnco_alpha,
            beta=cfg.rl.symnco_beta,
            num_augment=cfg.rl.num_augment,
            augment_fn=cfg.rl.augment_fn,
            num_starts=cfg.rl.num_starts,
            **common_kwargs,
        )
    elif cfg.rl.algorithm == "hrl":
        from logic.src.models.gat_lstm_manager import GATLSTManager

        manager = GATLSTManager(device=cfg.device, hidden_dim=cfg.rl.meta_hidden_dim)
        model = HRLModule(manager=manager, worker=policy, env=env, lr=cfg.rl.meta_lr)
    else:
        model = REINFORCE(baseline=cfg.rl.baseline, **common_kwargs)

    if getattr(cfg.rl, "use_meta", False):
        model = MetaRLModule(
            agent=model,
            meta_lr=cfg.rl.meta_lr,
            history_length=cfg.rl.meta_history_length,
            hidden_size=cfg.rl.meta_hidden_dim,
        )

    return model


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
                log=True if range_val[0] > 0 and range_val[1] / range_val[0] > 10 else False,
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
    )

    # 3. Train
    try:
        trainer.fit(model)
        return trainer.callback_metrics.get("val/reward", torch.tensor(float("-inf"))).item()
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return float("-inf")


def run_hpo(cfg: Config) -> float:
    """Run Hyperparameter Optimization."""
    if cfg.hpo.method == "dehb":
        from logic.src.pipeline.rl.hpo.dehb import DifferentialEvolutionHyperband

        def dehb_obj(config, fidelity):
            temp_cfg = OmegaConf.to_object(cfg)
            # Update config with suggested values
            for k, v in config.items():
                # Recursive attribute setting
                parts = k.split(".")
                obj = temp_cfg
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], v)

            model = create_model(temp_cfg)
            trainer = WSTrainer(max_epochs=int(fidelity), enable_progress_bar=False, logger=False)
            trainer.fit(model)
            # DEHB minimizes, so we return negative reward if maximizing
            reward = trainer.callback_metrics.get("val/reward", torch.tensor(0.0)).item()
            return {"fitness": -reward}

        dehb = DifferentialEvolutionHyperband(
            cs=cfg.hpo.search_space,
            f=dehb_obj,
            min_fidelity=cfg.hpo.min_epochs or 1,
            max_fidelity=cfg.hpo.n_epochs_per_trial,
        )
        best_config, runtime, _ = dehb.run(fevals=cfg.hpo.n_trials)
        logger.info(f"DEHB complete in {runtime:.2f}s. Best config: {best_config}")
        return dehb.get_incumbents()[1]

    # Default to Optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(),
    )

    logger.info(f"Starting {cfg.hpo.method} optimization with {cfg.hpo.n_trials} trials...")

    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=cfg.hpo.n_trials,
        n_jobs=1,
    )

    logger.info("Optimization complete!")
    logger.info(f"Best trial value: {study.best_value}")
    logger.info(f"Best parameters: {study.best_params}")
    return study.best_value


def run_training(cfg: Config) -> float:
    """Run single model training."""
    seed_everything(cfg.seed)
    model = create_model(cfg)

    trainer = WSTrainer(
        max_epochs=cfg.train.n_epochs,
        project_name="wsmart-route",
        experiment_name=cfg.experiment_name,
        accelerator=cfg.device if cfg.device != "cuda" else "auto",
        devices=1 if cfg.device == "cuda" else "auto",
        gradient_clip_val=float(cfg.rl.max_grad_norm) if cfg.rl.algorithm != "ppo" else 0.0,
        logger=False,
        callbacks=[SpeedMonitor(epoch_time=True)],
    )

    trainer.fit(model)
    return trainer.callback_metrics.get("val/reward", torch.tensor(0.0)).item()


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> float:
    """Unified entry point."""
    if cfg.hpo.n_trials > 0:
        return run_hpo(cfg)
    else:
        return run_training(cfg)


if __name__ == "__main__":
    main()
