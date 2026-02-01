"""
Unified Training and Hyperparameter Optimization entry point using PyTorch Lightning and Hydra.
"""

import os
from typing import Any, Dict, Tuple, Union, cast

import optuna
import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig, OmegaConf
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


def _remap_legacy_keys(common_kwargs: Dict[str, Any], cfg: Config) -> None:
    """Remap config keys to legacy names for simulation compatibility."""
    # The simulation's load_model expects legacy key names
    common_kwargs["problem"] = cfg.env.name
    common_kwargs["model"] = cfg.model.name
    common_kwargs["encoder"] = cfg.model.encoder_type
    common_kwargs["embed_dim"] = cfg.model.embed_dim
    common_kwargs["n_encode_layers"] = cfg.model.n_encode_layers
    common_kwargs["n_encode_sublayers"] = cfg.model.n_encoder_sublayers
    common_kwargs["n_decode_layers"] = cfg.model.n_decode_layers
    common_kwargs["n_heads"] = cfg.model.n_heads
    common_kwargs["n_predict_layers"] = cfg.model.n_predictor_layers
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
    # REMAPPING FOR SIMULATION COMPATIBILITY (args.json)
    _remap_legacy_keys(common_kwargs, cfg)

    # SANITIZATION: Ensure all values are primitives to satisfy YAML serialization
    # Recursively convert DictConfig/ListConfig/integers/floats/strings
    def deep_sanitize(obj: Any) -> Any:
        """
        Recursively convert OmegaConf objects and other types to primitive types.
        """
        if isinstance(obj, (DictConfig, ListConfig)):
            obj = OmegaConf.to_container(obj, resolve=True)

        if isinstance(obj, dict):
            return {str(k): deep_sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [deep_sanitize(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    # Sanitize common_kwargs before adding complex objects
    common_kwargs = cast(Dict[str, Any], deep_sanitize(common_kwargs))

    # Flatten nested algorithm specific configs back into common_kwargs for backward compatibility
    # with algorithm constructors that expect flat keys (like ppo_epochs, eps_clip, etc.)
    # Note: We prefix them if needed or just merge them.
    # For now, let's merge the active algorithm's config.
    algo_name = cfg.rl.algorithm
    if algo_name in common_kwargs and isinstance(common_kwargs[algo_name], dict):
        algo_specific = common_kwargs[algo_name]
        # Map modular names back to legacy names if needed, or just merge
        # PPO: epochs -> ppo_epochs, eps_clip -> eps_clip (already same)
        if algo_name == "ppo":
            common_kwargs["ppo_epochs"] = algo_specific.get("epochs", 10)
        elif algo_name == "sapo":
            common_kwargs["tau_pos"] = algo_specific.get("tau_pos", 0.1)
            common_kwargs["tau_neg"] = algo_specific.get("tau_neg", 1.0)
        # Merge all algorithm specific keys for general use
        common_kwargs.update(algo_specific)

    # Inject complex objects AFTER sanitization to avoid string conversion
    common_kwargs["env"] = env
    common_kwargs["policy"] = policy

    # Clean up common_kwargs to avoid passing unexpected args to LightningModule
    for key in ["lr_critic", "lr_critic_value"]:
        common_kwargs.pop(key, None)

    # 4. Define Algorithm Registry and Creation Helpers
    def _create_critic_helper(policy, cfg: Config) -> Any:
        from logic.src.models.policies.critic import create_critic_from_actor

        return create_critic_from_actor(
            policy,
            env_name=cfg.env.name,
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.n_encode_layers,
            n_heads=cfg.model.n_heads,
        )

    # Remove algorithm-specific arguments from common_kwargs to avoid duplicates when passed explicitly
    algorithm = cfg.rl.algorithm
    if algorithm in common_kwargs:
        common_kwargs.pop(algorithm)

    # Simplified Model Creation via Registry
    if algorithm in ["ppo", "sapo", "gspo", "dr_grpo"]:
        critic = _create_critic_helper(policy, cfg)
        algo_cls = {"ppo": PPO, "sapo": SAPO, "gspo": GSPO, "dr_grpo": DRGRPO}[algorithm]
        model: pl.LightningModule = algo_cls(critic=critic, **common_kwargs)

    elif algorithm == "gdpo":
        model = GDPO(
            gdpo_objective_keys=cfg.rl.gdpo.objective_keys,
            gdpo_objective_weights=cfg.rl.gdpo.objective_weights,
            gdpo_conditional_key=cfg.rl.gdpo.conditional_key,
            gdpo_renormalize=cfg.rl.gdpo.renormalize,
            **common_kwargs,
        )
    elif algorithm == "pomo":
        num_augment = cfg.rl.pomo.num_augment
        augment_fn = cfg.rl.pomo.augment_fn
        num_starts = cfg.rl.pomo.num_starts

        for k in ["num_augment", "augment_fn", "num_starts"]:
            common_kwargs.pop(k, None)

        model = POMO(
            num_augment=num_augment,
            augment_fn=augment_fn,
            num_starts=num_starts,
            **common_kwargs,
        )
    elif algorithm == "symnco":
        # Extract explicit args to avoid passing them twice through **common_kwargs
        alpha = cfg.rl.symnco.alpha
        beta = cfg.rl.symnco.beta
        num_augment = (
            cfg.rl.symnco.num_augment
            if hasattr(cfg.rl, "symnco") and hasattr(cfg.rl.symnco, "num_augment")
            else cfg.rl.pomo.num_augment
        )
        augment_fn = (
            cfg.rl.symnco.augment_fn
            if hasattr(cfg.rl, "symnco") and hasattr(cfg.rl.symnco, "augment_fn")
            else cfg.rl.pomo.augment_fn
        )
        num_starts = (
            cfg.rl.symnco.num_starts
            if hasattr(cfg.rl, "symnco") and hasattr(cfg.rl.symnco, "num_starts")
            else cfg.rl.pomo.num_starts
        )

        # Clean common_kwargs of these explicit params if they were flattened into it
        for k in ["alpha", "beta", "num_augment", "augment_fn", "num_starts"]:
            common_kwargs.pop(k, None)

        model = SymNCO(
            alpha=alpha,
            beta=beta,
            num_augment=num_augment,
            augment_fn=augment_fn,
            num_starts=num_starts,
            **common_kwargs,
        )
    elif algorithm == "hrl":
        from logic.src.models.gat_lstm_manager import GATLSTManager

        manager = GATLSTManager(device=cfg.device, hidden_dim=cfg.meta_rl.meta_hidden_dim)
        model = HRLModule(manager=manager, worker=policy, env=env, lr=cfg.meta_rl.meta_lr)
    elif algorithm in ["imitation", "adaptive_imitation"]:

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
            expert_kwargs: Dict[str, Any] = {"env_name": env_name}

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
                except (OSError, ValueError, KeyError) as e:
                    logger.warning(f"Failed to load {expert_name} config from {config_path}: {e}")

            # Specific legacy overrides if manual cfg.rl fields are present
            if expert_name in ["random_ls", "2opt"]:
                if "n_iterations" not in expert_kwargs:
                    expert_kwargs["n_iterations"] = int(getattr(cfg.rl.imitation, "random_ls_iterations", 100))
                if "op_probs" not in expert_kwargs:
                    expert_kwargs["op_probs"] = getattr(cfg.rl.imitation, "random_ls_op_probs", None)

            return expert_cls(**expert_kwargs)

        expert_name = cfg.rl.imitation.mode
        expert_policy = get_expert_policy(expert_name, cfg.env.name, cfg)

        if algorithm == "imitation":
            from logic.src.pipeline.rl.core.imitation import ImitationLearning

            model = ImitationLearning(expert_policy=expert_policy, expert_name=expert_name, **common_kwargs)
        else:  # adaptive_imitation
            from logic.src.pipeline.rl.core.adaptive_imitation import AdaptiveImitation

            model = AdaptiveImitation(
                expert_policy=expert_policy,
                il_weight=cfg.rl.adaptive_imitation.il_weight,
                il_decay=cfg.rl.adaptive_imitation.il_decay,
                patience=cfg.rl.adaptive_imitation.patience,
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
