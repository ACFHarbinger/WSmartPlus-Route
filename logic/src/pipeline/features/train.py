"""
Unified Training and Hyperparameter Optimization entry point using PyTorch Lightning and Hydra.
"""

import os
from collections.abc import MutableMapping
from typing import Any, Dict, Tuple, Union, cast

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
from logic.src.models.policies.classical.alns import VectorizedALNS
from logic.src.models.policies.classical.hgs import VectorizedHGS
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
from logic.src.pipeline.rl.common.trainer import WSTrainer
from logic.src.utils.configs.config_loader import load_yaml_config
from logic.src.utils.logging.pylogger import get_pylogger

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
        "alns": VectorizedALNS,
        "hgs": VectorizedHGS,
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
        heuristic = VectorizedALNS(env_name=cfg.env.name, max_iterations=500)
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
        common_kwargs = cast(Dict[str, Any], OmegaConf.to_container(OmegaConf.create(cast(Any, cfg.rl)), resolve=True))
    # Merge train, model and optim config into common_kwargs
    if isinstance(cfg.train, dict):
        train_params = cfg.train.copy()
    else:
        train_params = cast(
            Dict[str, Any], OmegaConf.to_container(OmegaConf.create(cast(Any, cfg.train)), resolve=True)
        )
    common_kwargs.update(train_params)

    if isinstance(cfg.model, dict):
        model_params = cfg.model.copy()
    else:
        model_params = cast(
            Dict[str, Any], OmegaConf.to_container(OmegaConf.create(cast(Any, cfg.model)), resolve=True)
        )
    common_kwargs.update(model_params)

    # We must remove 'name' as it's used in get_baseline and might conflict if passed in kwargs
    common_kwargs.pop("name", None)

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

    # REMAPPING FOR SIMULATION COMPATIBILITY (args.json)
    # The simulation's load_model expects legacy key names
    common_kwargs["problem"] = cfg.env.name
    common_kwargs["model"] = cfg.model.name
    common_kwargs["encoder"] = cfg.model.encoder_type
    common_kwargs["embedding_dim"] = cfg.model.embed_dim
    common_kwargs["n_encode_layers"] = cfg.model.num_encoder_layers
    common_kwargs["n_encode_sublayers"] = cfg.model.num_encoder_sublayers
    common_kwargs["n_decode_layers"] = cfg.model.num_decoder_layers
    common_kwargs["n_heads"] = cfg.model.num_heads
    common_kwargs["n_predict_layers"] = cfg.model.num_predictor_layers
    common_kwargs["learn_affine"] = cfg.model.learn_affine
    common_kwargs["track_stats"] = cfg.model.track_stats
    common_kwargs["epsilon_alpha"] = cfg.model.epsilon_alpha
    common_kwargs["momentum_beta"] = cfg.model.momentum_beta
    common_kwargs["af_param"] = cfg.model.activation_param
    common_kwargs["af_threshold"] = cfg.model.activation_threshold
    common_kwargs["af_replacement"] = cfg.model.activation_replacement
    common_kwargs["af_nparams"] = cfg.model.activation_num_parameters
    common_kwargs["af_urange"] = cfg.model.activation_uniform_range
    common_kwargs["aggregation"] = cfg.model.aggregation_node
    common_kwargs["aggregation_graph"] = cfg.model.aggregation_graph
    common_kwargs["hidden_dim"] = cfg.model.hidden_dim

    # Specific remapping if needed
    common_kwargs["optimizer"] = cfg.optim.optimizer
    common_kwargs["optimizer_kwargs"] = {
        "lr": cfg.optim.lr,
        "weight_decay": cfg.optim.weight_decay,
    }
    common_kwargs["lr_scheduler"] = cfg.optim.lr_scheduler

    # SANITIZATION: Ensure all values are primitives to satisfy YAML serialization
    # Recursively convert DictConfig/ListConfig/integers/floats/strings
    from omegaconf import DictConfig, ListConfig

    def deep_sanitize(obj):
        """
        Recursively convert OmegaConf objects to primitive types (dict/list).

        Args:
            obj: The configuration object (DictConfig, ListConfig, or other).

        Returns:
            The sanitized object (dict, list, or primitive).
        """
        if isinstance(obj, (dict, MutableMapping, DictConfig)):
            # Force conversion to dict if it's OmegaConf
            if isinstance(obj, DictConfig):
                obj = OmegaConf.to_container(obj, resolve=True)
            return {k: deep_sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, ListConfig)):
            # Force conversion to list if it's OmegaConf
            if isinstance(obj, ListConfig):
                obj = OmegaConf.to_container(obj, resolve=True)
            return [deep_sanitize(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        # Fallback for objects that might be convertible to str
        return str(obj)

    # Sanitize common_kwargs before adding complex objects
    common_kwargs = cast(Dict[str, Any], deep_sanitize(common_kwargs))

    # Inject complex objects AFTER sanitization to avoid string conversion
    common_kwargs["env"] = env
    common_kwargs["policy"] = policy

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
    elif cfg.rl.algorithm in ["imitation", "adaptive_imitation"]:
        # Helper to load expert policy with custom config
        def get_expert_policy(expert_name: str, env_name: str, cfg: Config) -> Any:
            expert_map = {
                "hgs": VectorizedHGS,
                "alns": VectorizedALNS,
                "random_ls": RandomLocalSearchPolicy,
                "2opt": RandomLocalSearchPolicy,
            }
            if expert_name not in expert_map:
                raise ValueError(f"Unknown expert: {expert_name}")

            expert_cls = expert_map[expert_name]
            expert_kwargs = {"env_name": env_name}

            # Strategy: Load from model.policy_config OR default path
            config_path = getattr(cfg.model, "policy_config", None)
            if config_path is None:
                default_path = f"scripts/configs/model/{expert_name}.yaml"
                if os.path.exists(default_path):
                    config_path = default_path

            if config_path and os.path.exists(config_path):
                try:
                    custom_params = load_yaml_config(config_path)
                    if custom_params:
                        expert_kwargs.update(custom_params)
                        logger.info(f"Loaded {expert_name} configuration from {config_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {expert_name} config from {config_path}: {e}")

            # Specific legacy overrides if manual cfg.rl fields are present
            if expert_name in ["random_ls", "2opt"]:
                if "n_iterations" not in expert_kwargs:
                    expert_kwargs["n_iterations"] = getattr(cfg.rl, "random_ls_iterations", 100)
                if "op_probs" not in expert_kwargs:
                    expert_kwargs["op_probs"] = getattr(cfg.rl, "random_ls_op_probs", None)

            return expert_cls(**expert_kwargs)

        expert_name = getattr(cfg.rl, "imitation_mode", "hgs")
        expert_policy = get_expert_policy(expert_name, cfg.env.name, cfg)

        if cfg.rl.algorithm == "imitation":
            from logic.src.pipeline.rl.core.imitation import ImitationLearning

            model = ImitationLearning(expert_policy=expert_policy, expert_name=expert_name, **common_kwargs)
        else:  # adaptive_imitation
            from logic.src.pipeline.rl.core.adaptive_imitation import AdaptiveImitation

            model = AdaptiveImitation(
                expert_policy=expert_policy,
                il_weight=getattr(cfg.rl, "il_weight", 1.0),
                il_decay=getattr(cfg.rl, "il_decay", 0.95),
                patience=getattr(cfg.rl, "patience", 5),
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
            cs=cast(Dict[str, Union[Tuple[float, float], list]], cfg.hpo.search_space),
            f=dehb_obj,
            min_fidelity=getattr(cfg.hpo, "min_epochs", 1) or 1,
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
        logger=CSVLogger(cfg.train.logs_dir or "logs", name=""),
        callbacks=[SpeedMonitor(epoch_time=True)],
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_step,
        model_weights_path=cfg.train.model_weights_path,
        logs_dir=cfg.train.logs_dir,
        reload_dataloaders_every_n_epochs=cfg.train.reload_dataloaders_every_n_epochs,
    )

    trainer.fit(model)

    # Save final weights if path is provided
    if cfg.train.final_model_path:
        model.save_weights(cfg.train.final_model_path)

    return trainer.callback_metrics.get("val/reward", torch.tensor(0.0)).item()


@hydra.main(version_base=None, config_path="../../../../scripts/configs", config_name="config")
def main(cfg: Config) -> float:
    """Unified entry point."""
    if cfg.task == "train":
        if cfg.hpo.n_trials > 0:
            return run_hpo(cfg)
        else:
            return run_training(cfg)
    elif cfg.task == "eval":
        from omegaconf import OmegaConf

        from logic.src.pipeline.features.eval import run_evaluate_model, validate_eval_args

        # Convert Hydra config to dict
        eval_args = OmegaConf.to_container(cfg.eval, resolve=True)
        # Validate and run
        args = validate_eval_args(eval_args)
        run_evaluate_model(args)
        return 0.0
    elif cfg.task == "test_sim":
        from omegaconf import OmegaConf

        from logic.src.pipeline.features.test import run_wsr_simulator_test, validate_test_sim_args

        # Convert Hydra config to dict
        sim_args = OmegaConf.to_container(cfg.sim, resolve=True)
        # Validate and run
        args = validate_test_sim_args(sim_args)
        run_wsr_simulator_test(args)
        return 0.0
    elif cfg.task == "gen_data":
        from omegaconf import OmegaConf

        from logic.src.data.generate_data import generate_datasets, validate_gen_data_args

        # Convert Hydra config to dict
        data_args = OmegaConf.to_container(cfg.data, resolve=True)
        # Validate and run
        args = validate_gen_data_args(data_args)
        generate_datasets(args)
        return 0.0
    else:
        raise ValueError(f"Unknown task: {cfg.task}")


if __name__ == "__main__":
    main()
