from typing import Any, Dict, cast

import pytorch_lightning as pl
from omegaconf import OmegaConf

from logic.src.configs import Config
from logic.src.envs import get_env
from logic.src.models.policies import (
    AttentionModelPolicy,
    DeepDecoderPolicy,
    MoEPolicy,
    PointerNetworkPolicy,
    SymNCOPolicy,
    TemporalAMPolicy,
)
from logic.src.models.policies.adaptive_large_neighborhood_search import VectorizedALNS
from logic.src.models.policies.hgs import VectorizedHGS
from logic.src.models.policies.hgs_alns import VectorizedHGSALNS
from logic.src.models.policies.hybrid import NeuralHeuristicHybrid
from logic.src.pipeline.features.base import deep_sanitize, remap_legacy_keys
from logic.src.pipeline.rl import REINFORCE, MetaRLModule
from logic.src.utils.logging.pylogger import get_pylogger

from .registry import _ALGO_REGISTRY

logger = get_pylogger(__name__)


def create_model(cfg: Config) -> pl.LightningModule:
    """Helper to create the RL model based on config."""
    # 1. Initialize Environment
    env_name = cfg.env.name
    env_kwargs = {k: v for k, v in vars(cfg.env).items() if k != "name"}
    # Override distribution if specified in train config
    if getattr(cfg.train, "data_distribution", None) is not None:
        env_kwargs["data_distribution"] = cfg.train.data_distribution
    elif "data_distribution" not in env_kwargs or env_kwargs["data_distribution"] is None:
        env_kwargs["data_distribution"] = "unif"

    env_kwargs["device"] = cfg.device
    env_kwargs["batch_size"] = cfg.train.batch_size
    env = get_env(env_name, **env_kwargs)

    # 2. Initialize Policy
    policy_map = {
        "am": AttentionModelPolicy,
        "moe": MoEPolicy,
        "deep_decoder": DeepDecoderPolicy,
        "temporal": TemporalAMPolicy,
        "pointer": PointerNetworkPolicy,
        "symnco": SymNCOPolicy,
        "alns": VectorizedALNS,
        "hgs": VectorizedHGS,
        "hgs_alns": VectorizedHGSALNS,
        "hybrid": NeuralHeuristicHybrid,
    }

    if cfg.model.name not in policy_map:
        raise ValueError(f"Unknown model name: {cfg.model.name}. Available: {list(policy_map.keys())}")

    policy_cls = policy_map[cfg.model.name]
    policy_kwargs = vars(cfg.model).copy()
    policy_kwargs["env_name"] = cfg.env.name

    for key in ["lr_critic", "lr_critic_value"]:
        policy_kwargs.pop(key, None)
    policy_kwargs.pop("name", None)
    if cfg.model.name == "hybrid":
        neural = AttentionModelPolicy(**policy_kwargs)
        heuristic = VectorizedALNS(env_name=cfg.env.name, max_iterations=500)
        policy = NeuralHeuristicHybrid(neural, heuristic)
    else:
        policy = policy_cls(**policy_kwargs)

    # 3. Initialize RL Module
    if isinstance(cfg.rl, dict):
        common_kwargs = cfg.rl.copy()
    else:
        common_kwargs = cast(Dict[str, Any], OmegaConf.to_container(OmegaConf.create(cast(Any, cfg.rl)), resolve=True))

    if isinstance(cfg.train, dict):
        train_params = cfg.train.copy()
    else:
        train_params = cast(
            Dict[str, Any], OmegaConf.to_container(OmegaConf.create(cast(Any, cfg.train)), resolve=True)
        )
    common_kwargs.update(train_params)

    if getattr(cfg.train, "load_dataset", None) is not None:
        common_kwargs["train_dataset_path"] = cfg.train.load_dataset
    elif getattr(cfg.train, "train_dataset", None) is not None:
        common_kwargs["train_dataset_path"] = cfg.train.train_dataset

    if "val_dataset" in common_kwargs:
        common_kwargs["val_dataset_path"] = common_kwargs.pop("val_dataset")

    if isinstance(cfg.model, dict):
        model_params = cfg.model.copy()
    else:
        model_params = cast(
            Dict[str, Any], OmegaConf.to_container(OmegaConf.create(cast(Any, cfg.model)), resolve=True)
        )
    common_kwargs.update(model_params)
    common_kwargs.pop("name", None)

    common_kwargs["env"] = env
    common_kwargs["policy"] = policy
    common_kwargs["optimizer"] = cfg.optim.optimizer
    common_kwargs["optimizer_kwargs"] = {
        "lr": cfg.optim.lr,
        "weight_decay": cfg.optim.weight_decay,
    }
    common_kwargs["lr_scheduler"] = cfg.optim.lr_scheduler

    scheduler_kwargs = cfg.optim.lr_scheduler_kwargs.copy()
    if cfg.optim.lr_decay != 1.0:
        scheduler_kwargs["gamma"] = cfg.optim.lr_decay
    if cfg.optim.lr_min_value != 0.0:
        scheduler_kwargs["eta_min"] = cfg.optim.lr_min_value

    common_kwargs["lr_scheduler_kwargs"] = scheduler_kwargs
    common_kwargs["baseline"] = cfg.rl.baseline

    remap_legacy_keys(common_kwargs, cfg)
    common_kwargs = cast(Dict[str, Any], deep_sanitize(common_kwargs))

    algo_name = cfg.rl.algorithm
    if algo_name in common_kwargs and isinstance(common_kwargs[algo_name], dict):
        algo_specific = common_kwargs[algo_name]
        if algo_name == "ppo":
            common_kwargs["ppo_epochs"] = algo_specific.get("epochs", 10)
        elif algo_name == "sapo":
            common_kwargs["tau_pos"] = algo_specific.get("tau_pos", 0.1)
            common_kwargs["tau_neg"] = algo_specific.get("tau_neg", 1.0)
        common_kwargs.update(algo_specific)

    common_kwargs["env"] = env
    common_kwargs["policy"] = policy

    for key in ["lr_critic", "lr_critic_value"]:
        common_kwargs.pop(key, None)

    must_go_selector = None
    if hasattr(cfg, "must_go") and cfg.must_go is not None:
        from logic.src.policies.selection import create_selector_from_config

        must_go_selector = create_selector_from_config(cfg.must_go)
        if must_go_selector is not None:
            logger.info(f"Must-go selector created: {cfg.must_go.strategy}")
    common_kwargs["must_go_selector"] = must_go_selector

    # Algorithm Factory
    factory = _ALGO_REGISTRY.get(algo_name)
    if factory is not None:
        model: pl.LightningModule = factory(cfg, policy, env, common_kwargs)
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
