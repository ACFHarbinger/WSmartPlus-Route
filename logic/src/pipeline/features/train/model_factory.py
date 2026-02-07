"""
Model factory for training pipeline.
"""

import os
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
from logic.src.models.policies.classical.alns import VectorizedALNS
from logic.src.models.policies.classical.hgs import VectorizedHGS
from logic.src.models.policies.classical.hgs_alns import VectorizedHGSALNS
from logic.src.models.policies.classical.hybrid import NeuralHeuristicHybrid
from logic.src.models.policies.classical.random_local_search import (
    RandomLocalSearchPolicy,
)
from logic.src.pipeline.features.base import deep_sanitize, remap_legacy_keys
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
from logic.src.utils.configs.config_loader import load_yaml_config
from logic.src.utils.logging.pylogger import get_pylogger

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


def _create_ppo_family(algo_name: str, cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    critic = _create_critic_helper(policy, cfg)
    cls_map = {"ppo": PPO, "sapo": SAPO, "gspo": GSPO, "dr_grpo": DRGRPO}
    return cls_map[algo_name](critic=critic, **kw)


def _create_gdpo(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    return GDPO(
        gdpo_objective_keys=cfg.rl.gdpo.objective_keys,
        gdpo_objective_weights=cfg.rl.gdpo.objective_weights,
        gdpo_conditional_key=cfg.rl.gdpo.conditional_key,
        gdpo_renormalize=cfg.rl.gdpo.renormalize,
        **kw,
    )


def _create_pomo(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    explicit = {
        "num_augment": cfg.rl.pomo.num_augment,
        "augment_fn": cfg.rl.pomo.augment_fn,
        "num_starts": cfg.rl.pomo.num_starts,
    }
    for k in explicit:
        kw.pop(k, None)
    return POMO(
        num_augment=int(cast(Any, explicit["num_augment"])),
        augment_fn=explicit["augment_fn"],  # type: ignore
        num_starts=int(cast(Any, explicit["num_starts"])) if explicit["num_starts"] is not None else None,
        **kw,
    )


def _create_symnco(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    explicit = {
        "alpha": cfg.rl.symnco.alpha,
        "beta": cfg.rl.symnco.beta,
        "num_augment": (
            cfg.rl.symnco.num_augment
            if hasattr(cfg.rl, "symnco") and hasattr(cfg.rl.symnco, "num_augment")
            else cfg.rl.pomo.num_augment
        ),
        "augment_fn": (
            cfg.rl.symnco.augment_fn
            if hasattr(cfg.rl, "symnco") and hasattr(cfg.rl.symnco, "augment_fn")
            else cfg.rl.pomo.augment_fn
        ),
        "num_starts": (
            cfg.rl.symnco.num_starts
            if hasattr(cfg.rl, "symnco") and hasattr(cfg.rl.symnco, "num_starts")
            else cfg.rl.pomo.num_starts
        ),
    }
    for k in explicit:
        kw.pop(k, None)
    return SymNCO(
        alpha=float(cast(Any, explicit["alpha"])) if explicit["alpha"] is not None else 1.0,
        beta=float(cast(Any, explicit["beta"])) if explicit["beta"] is not None else 1.0,
        num_augment=int(cast(Any, explicit["num_augment"])),
        augment_fn=explicit["augment_fn"],  # type: ignore
        num_starts=int(cast(Any, explicit["num_starts"])) if explicit["num_starts"] is not None else None,
        **kw,
    )


def _create_hrl(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    from logic.src.models.gat_lstm_manager import GATLSTManager

    manager = GATLSTManager(device=cfg.device, hidden_dim=cfg.meta_rl.meta_hidden_dim)
    return HRLModule(manager=manager, worker=policy, env=env, lr=cfg.meta_rl.meta_lr)


def _get_expert_policy(expert_name: str, env_name: str, cfg: Config) -> Any:
    expert_map = {
        "hgs": VectorizedHGS,
        "alns": VectorizedALNS,
        "hgs_alns": VectorizedHGSALNS,
        "random_ls": RandomLocalSearchPolicy,
    }
    if expert_name not in expert_map:
        raise ValueError(f"Unknown expert: {expert_name}")

    expert_cls = expert_map[expert_name]
    expert_kwargs: Dict[str, Any] = {"env_name": env_name}

    config_path = getattr(cfg.model, "policy_config", None)
    if config_path is None:
        default_path = f"assets/configs/model/{expert_name}.yaml"
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

    if expert_name in ["random_ls", "2opt"]:
        if "n_iterations" not in expert_kwargs:
            expert_kwargs["n_iterations"] = int(getattr(cfg.rl.imitation, "random_ls_iterations", 100))
        if "op_probs" not in expert_kwargs:
            expert_kwargs["op_probs"] = getattr(cfg.rl.imitation, "random_ls_op_probs", None)

    return expert_cls(**expert_kwargs)


def _create_imitation(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    from logic.src.pipeline.rl.core.imitation import ImitationLearning

    expert_policy = _get_expert_policy(cfg.rl.imitation.mode, cfg.env.name, cfg)
    return ImitationLearning(expert_policy=expert_policy, expert_name=cfg.rl.imitation.mode, **kw)


def _create_adaptive_imitation(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    from logic.src.pipeline.rl.core.adaptive_imitation import AdaptiveImitation

    expert_policy = _get_expert_policy(cfg.rl.imitation.mode, cfg.env.name, cfg)
    return AdaptiveImitation(expert_policy=expert_policy, **kw)


_ALGO_REGISTRY: Dict[str, Any] = {
    "ppo": lambda c, p, e, kw: _create_ppo_family("ppo", c, p, e, kw),
    "sapo": lambda c, p, e, kw: _create_ppo_family("sapo", c, p, e, kw),
    "gspo": lambda c, p, e, kw: _create_ppo_family("gspo", c, p, e, kw),
    "dr_grpo": lambda c, p, e, kw: _create_ppo_family("dr_grpo", c, p, e, kw),
    "gdpo": _create_gdpo,
    "pomo": _create_pomo,
    "symnco": _create_symnco,
    "hrl": _create_hrl,
    "imitation": _create_imitation,
    "adaptive_imitation": _create_adaptive_imitation,
    "reinforce": lambda c, p, e, kw: REINFORCE(**kw),
}
