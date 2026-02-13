"""builder.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import builder
"""

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, cast

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from logic.src.configs import Config
from logic.src.envs import get_env
from logic.src.interfaces import ITraversable
from logic.src.models.policies import (
    AttentionModelPolicy,
    DeepDecoderPolicy,
    MoEPolicy,
    NeuralHeuristicHybrid,
    PointerNetworkPolicy,
    SymNCOPolicy,
    TemporalAMPolicy,
    VectorizedALNS,
    VectorizedHGS,
    VectorizedHGSALNS,
)
from logic.src.pipeline.features.base import deep_sanitize, remap_legacy_keys
from logic.src.pipeline.rl import REINFORCE, MetaRLModule
from logic.src.utils.logging.pylogger import get_pylogger

from .registry import _ALGO_REGISTRY

logger = get_pylogger(__name__)


def create_model(cfg: Config) -> pl.LightningModule:
    """Helper to create the RL model based on config."""
    # 1. Initialize Environment
    env = _init_environment(cfg)

    # 2. Initialize Policy
    policy = _init_policy(cfg, env)

    # 3. Initialize RL Module Keyword Arguments
    common_kwargs = _prepare_rl_kwargs(cfg, env, policy)

    # 4. Algorithm Factory
    algo_name = cfg.rl.algorithm
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


def _init_environment(cfg: Config):
    """Initialize the environment based on config."""
    env_name = cfg.env.name
    env_kwargs = {k: v for k, v in vars(cfg.env).items() if k not in ["name", "graph", "reward"]}

    # Flatten GraphConfig
    if hasattr(cfg.env, "graph"):
        graph = cfg.env.graph
        env_kwargs.update(
            {
                "area": graph.area,
                "waste_type": graph.waste_type,
                "vertex_method": graph.vertex_method,
                "distance_method": graph.distance_method,
                "dm_filepath": graph.dm_filepath,
                "edge_threshold": graph.edge_threshold,
                "edge_method": graph.edge_method,
                "focus_graph": graph.focus_graph,
                "focus_size": graph.focus_size,
                "eval_focus_size": graph.eval_focus_size,
            }
        )

    # Flatten ObjectiveConfig
    if hasattr(cfg.env, "reward"):
        reward = cfg.env.reward
        env_kwargs.update(
            {
                "cost_weight": reward.cost_weight,
                "waste_weight": reward.waste_weight,
                "overflow_penalty": reward.overflow_penalty,
            }
        )

    # Override distribution if specified in train config
    dist = getattr(cfg.train, "data_distribution", None)
    env_kwargs["data_distribution"] = dist if dist is not None else (env_kwargs.get("data_distribution") or "unif")
    env_kwargs["device"] = cfg.device
    env_kwargs["batch_size"] = cfg.train.batch_size

    return get_env(env_name, **env_kwargs)


def _init_policy(cfg: Config, env: Any):
    """Initialize the policy based on config."""
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

    if cfg.model.name == "hybrid":
        return _init_hybrid_policy(cfg)

    # Flatten ModelConfig for policy initialization
    policy_kwargs: Dict[str, Any] = {"env_name": cfg.env.name}

    if hasattr(cfg.model, "encoder"):
        enc = cfg.model.encoder
        policy_kwargs.update(
            {
                "embed_dim": enc.embed_dim,
                "hidden_dim": enc.hidden_dim,
                "n_encode_layers": enc.n_layers,
                "n_heads": enc.n_heads,
                "normalization": enc.normalization.norm_type if hasattr(enc, "normalization") else "instance",
                "activation": enc.activation.name if hasattr(enc, "activation") else "relu",
                "norm_config": enc.normalization,
                "activation_config": enc.activation,
                "encoder_config": enc,
            }
        )

    if hasattr(cfg.model, "decoder"):
        dec = cfg.model.decoder
        policy_kwargs.update(
            {
                "n_decode_layers": dec.n_layers,
                "decoder_type": dec.type,
                "decoder_config": dec,
            }
        )

    # Copy other top-level model params
    for k, v in vars(cfg.model).items():
        if k not in ["encoder", "decoder", "name"]:
            policy_kwargs[k] = v

    for key in ["lr_critic", "lr_critic_value"]:
        policy_kwargs.pop(key, None)

    return policy_map[cfg.model.name](**policy_kwargs)


def _init_hybrid_policy(cfg: Config):
    """Special handling for hybrid construction-refinement policy."""
    # 1. Neural construction policy
    neural_cfg = cast(Any, cfg).copy()  # Simplified copy
    neural_cfg.model.name = "am"
    neural_policy = create_model(cast(Config, neural_cfg)).policy

    # 2. Heuristic refinement policy
    ref_strategy = getattr(cfg.rl, "refinement_strategy", "alns")
    ref_time = getattr(cfg.rl, "refinement_time_limit", 5.0)
    ref_iters = getattr(cfg.rl, "refinement_iterations", 500)
    max_v = getattr(cfg.rl, "max_vehicles", 0)

    if ref_strategy == "alns":
        heuristic_policy = VectorizedALNS(
            env_name=cfg.env.name, time_limit=ref_time, max_iterations=ref_iters, max_vehicles=max_v
        )
    else:
        heuristic_policy = VectorizedHGS(env_name=cfg.env.name, time_limit=ref_time, max_iterations=ref_iters)  # type: ignore[assignment]
    return NeuralHeuristicHybrid(neural_policy=neural_policy, heuristic_policy=heuristic_policy)


def _config_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert config object (dataclass or DictConfig) to dict."""
    if is_dataclass(obj):
        return asdict(obj)  # type: ignore[arg-type]
    elif isinstance(obj, DictConfig):
        return cast(Dict[str, Any], OmegaConf.to_container(obj, resolve=True))
    elif isinstance(obj, dict):
        return obj
    else:
        # Fallback: try to wrap in OmegaConf.create and convert
        return cast(Dict[str, Any], OmegaConf.to_container(OmegaConf.create(obj), resolve=True))


def _prepare_rl_kwargs(cfg: Config, env: Any, policy: Any):
    """Prepare keyword arguments for RL module initialization."""
    # Prepare base dicts
    common_kwargs: Dict[str, Any] = _config_to_dict(cfg.rl)

    train_params = cast(Any, cfg.train).copy() if hasattr(cfg.train, "copy") else _config_to_dict(cfg.train)
    common_kwargs.update(train_params)

    # Dataset paths
    train_path = getattr(cfg.train, "load_dataset", getattr(cfg.train, "train_dataset", None))
    if train_path:
        common_kwargs["train_dataset_path"] = train_path
    if "val_dataset" in common_kwargs:
        common_kwargs["val_dataset_path"] = common_kwargs.pop("val_dataset")

    # Model params
    model_params = cast(Any, cfg.model).copy() if hasattr(cfg.model, "copy") else _config_to_dict(cfg.model)
    common_kwargs.update(model_params)
    common_kwargs.pop("name", None)

    # Inject core objects
    common_kwargs.update(
        {
            "env": env,
            "policy": policy,
            "optimizer": cfg.optim.optimizer,
            "optimizer_kwargs": {"lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay},
            "lr_scheduler": cfg.optim.lr_scheduler,
        }
    )

    # Scheduler config
    sch_kwargs = cfg.optim.lr_scheduler_kwargs.copy()
    if cfg.optim.lr_decay != 1.0:
        sch_kwargs["gamma"] = cfg.optim.lr_decay
    if cfg.optim.lr_min_value != 0.0:
        sch_kwargs["eta_min"] = cfg.optim.lr_min_value
    common_kwargs["lr_scheduler_kwargs"] = sch_kwargs
    common_kwargs["baseline"] = cfg.rl.baseline

    remap_legacy_keys(common_kwargs, cfg)
    common_kwargs = cast(Dict[str, Any], deep_sanitize(common_kwargs))

    # Algorithm specific overrides
    algo_name = cfg.rl.algorithm
    if algo_name in common_kwargs and isinstance(common_kwargs[algo_name], (dict, ITraversable)):
        algo_specific = common_kwargs[algo_name]
        if algo_name == "ppo":
            common_kwargs["ppo_epochs"] = algo_specific.get("epochs", 10)
        elif algo_name == "sapo":
            common_kwargs.update(
                {"tau_pos": algo_specific.get("tau_pos", 0.1), "tau_neg": algo_specific.get("tau_neg", 1.0)}
            )
        common_kwargs.update(algo_specific)

    for key in ["lr_critic", "lr_critic_value"]:
        common_kwargs.pop(key, None)

    # Must-go selector
    must_go_selector = None
    if hasattr(cfg, "must_go") and cfg.must_go is not None:
        from logic.src.policies.other.must_go import create_selector_from_config

        must_go_selector = create_selector_from_config(cfg.must_go)
        if must_go_selector:
            logger.info(f"Must-go selector created: {cfg.must_go.strategy}")
    common_kwargs["must_go_selector"] = must_go_selector

    return common_kwargs
