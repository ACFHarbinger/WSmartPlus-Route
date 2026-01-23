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
from logic.src.models.policies.classical.random_local_search import (
    RandomLocalSearchPolicy,
)
from logic.src.pipeline.rl import (
    DRGRPO,
    GDPO,
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
    env_kwargs["device"] = cfg.device  # Pass device to environment
    env_kwargs["batch_size"] = cfg.train.batch_size  # Match training batch size
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
    policy_kwargs = vars(cfg.model).copy()
    policy_kwargs["env_name"] = cfg.env.name

    # Mapping for backward compatibility with existing policy classes
    if "num_encoder_layers" in policy_kwargs:
        policy_kwargs["n_encode_layers"] = policy_kwargs.pop("num_encoder_layers")
    if "num_decoder_layers" in policy_kwargs:
        policy_kwargs["n_decode_layers"] = policy_kwargs.pop("num_decoder_layers")
    if "num_heads" in policy_kwargs:
        policy_kwargs["n_heads"] = policy_kwargs.pop("num_heads")

    # Remove fields not used in policy __init__ if needed, or rely on **kwargs
    for key in ["lr_critic", "lr_critic_value"]:
        policy_kwargs.pop(key, None)
    policy_kwargs.pop("name", None)
    if cfg.model.name == "hybrid":
        neural = AttentionModelPolicy(**policy_kwargs)
        heuristic = ALNSPolicy(env_name=cfg.env.name)
        policy = NeuralHeuristicHybrid(neural, heuristic)
    else:
        # Some policies like ALNSPolicy/HGSPolicy might take more specific args
        # For now, pass what we have in ModelConfig
        policy = policy_cls(**policy_kwargs)

    # 3. Initialize RL Module
    # Convert to primitive dict using OmegaConf to avoid 'Unions of containers' errors during checkpointing
    # This strips types like List[Any] that cause issues with save_hyperparameters
    if isinstance(cfg.rl, dict):
        common_kwargs = cfg.rl.copy()
    else:
        # Works for both Dataclass and DictConfig
        common_kwargs = OmegaConf.to_container(OmegaConf.create(cfg.rl), resolve=True)
    # Merge train and optim config into common_kwargs
    if isinstance(cfg.train, dict):
        train_params = cfg.train.copy()
    else:
        train_params = OmegaConf.to_container(OmegaConf.create(cfg.train), resolve=True)
    common_kwargs.update(train_params)

    # Specific remapping if needed
    common_kwargs["env"] = env
    common_kwargs["policy"] = policy
    common_kwargs["optimizer"] = cfg.optim.optimizer
    common_kwargs["optimizer_kwargs"] = {
        "lr": cfg.optim.lr,
        "weight_decay": cfg.optim.weight_decay,
    }
    common_kwargs["lr_scheduler"] = cfg.optim.lr_scheduler

    # Merge scheduler kwargs if any
    scheduler_kwargs = cfg.optim.lr_scheduler_kwargs.copy()
    # Map explicit scheduler fields to kwargs if they are not None/default
    if cfg.optim.lr_decay != 1.0:
        scheduler_kwargs["gamma"] = cfg.optim.lr_decay
    if cfg.optim.lr_min_value != 0.0:
        scheduler_kwargs["eta_min"] = cfg.optim.lr_min_value

    common_kwargs["lr_scheduler_kwargs"] = scheduler_kwargs
    common_kwargs["baseline"] = cfg.rl.baseline

    # Clean up common_kwargs to avoid passing unexpected args to LightningModule
    # some algorithm specific args might be in cfg.rl but not in common_kwargs base
    for key in ["lr_critic", "lr_critic_value"]:
        common_kwargs.pop(key, None)

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
    elif cfg.rl.algorithm == "gdpo":
        model = GDPO(
            gdpo_objective_keys=cfg.rl.gdpo_objective_keys,
            gdpo_objective_weights=cfg.rl.gdpo_objective_weights,
            gdpo_conditional_key=cfg.rl.gdpo_conditional_key,
            gdpo_renormalize=cfg.rl.gdpo_renormalize,
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

        manager = GATLSTManager(device=cfg.device, hidden_dim=cfg.meta_rl.meta_hidden_dim)
        model = HRLModule(manager=manager, worker=policy, env=env, lr=cfg.meta_rl.meta_lr)
    elif cfg.rl.algorithm == "imitation":
        from logic.src.pipeline.rl.core.imitation import ImitationLearning

        # Determine expert
        expert_name = cfg.rl.get("imitation_mode", "hgs")
        expert_policy = None
        if expert_name == "hgs":
            expert_policy = HGSPolicy(env_name=cfg.env.name)
        elif expert_name == "alns":
            expert_policy = ALNSPolicy(env_name=cfg.env.name)
        elif expert_name in ["random_ls", "2opt"]:
            expert_policy = RandomLocalSearchPolicy(
                env_name=cfg.env.name,
                n_iterations=cfg.rl.random_ls_iterations,
                op_probs=cfg.rl.random_ls_op_probs,
            )

        model = ImitationLearning(expert_policy=expert_policy, expert_name=expert_name, **common_kwargs)
    elif cfg.rl.algorithm == "adaptive_imitation":
        from logic.src.pipeline.rl.core.adaptive_imitation import AdaptiveImitation

        expert_name = cfg.rl.get("imitation_mode", "hgs")
        expert_policy = None
        if expert_name == "hgs":
            expert_policy = HGSPolicy(env_name=cfg.env.name)
        elif expert_name == "alns":
            expert_policy = ALNSPolicy(env_name=cfg.env.name)
        elif expert_name in ["random_ls", "2opt"]:
            expert_policy = RandomLocalSearchPolicy(
                env_name=cfg.env.name,
                n_iterations=cfg.rl.random_ls_iterations,
                op_probs=cfg.rl.random_ls_op_probs,
            )

        model = AdaptiveImitation(
            expert_policy=expert_policy,
            il_weight=cfg.rl.get("il_weight", 1.0),
            il_decay=cfg.rl.get("il_decay", 0.95),
            patience=cfg.rl.get("patience", 5),
            **common_kwargs,
        )
    else:
        model = REINFORCE(**common_kwargs)

    if getattr(cfg.meta_rl, "use_meta", False):
        model = MetaRLModule(
            agent=model,
            meta_lr=cfg.meta_rl.meta_lr,
            history_length=cfg.meta_rl.meta_history_length,
            hidden_size=cfg.meta_rl.meta_hidden_dim,
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
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Trial failed: {e}")
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
            """
            DEHB objective function.

            Args:
                config: Hyperparameter configuration dict.
                fidelity: Training fidelity (epochs).

            Returns:
                dict: Fitness dictionary with negative reward.
            """
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
            trainer = WSTrainer(
                max_epochs=int(fidelity),
                enable_progress_bar=False,
                logger=False,
                log_every_n_steps=temp_cfg.train.log_step,
            )
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
        # Return best value (inverted back if needed, but dehb returns config)
        # We can re-evaluate or just return placeholder
        return 0.0

    # 2. Optuna Methods (TPE, Grid, Random, Hyperband)
    hpo_runner = OptunaHPO(cfg, objective)
    return hpo_runner.run()


def run_training(cfg: Config) -> float:
    """Run single model training."""
    seed_everything(cfg.seed)

    # Enable Tensor Core acceleration for Ampere+ GPUs
    if torch.cuda.is_available() and cfg.train.precision in ["16-mixed", "bf16-mixed"]:
        torch.set_float32_matmul_precision("medium")

    model = create_model(cfg)

    from pytorch_lightning.loggers import CSVLogger

    trainer = WSTrainer(
        max_epochs=cfg.train.n_epochs,
        project_name="wsmart-route",
        experiment_name=cfg.experiment_name,
        accelerator=cfg.device if cfg.device != "cuda" else "auto",
        devices=1 if cfg.device == "cuda" else "auto",
        gradient_clip_val=(float(cfg.rl.max_grad_norm) if cfg.rl.algorithm != "ppo" else 0.0),
        logger=CSVLogger("logs", name="lightning_logs"),
        callbacks=[SpeedMonitor(epoch_time=True)],
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_step,
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
