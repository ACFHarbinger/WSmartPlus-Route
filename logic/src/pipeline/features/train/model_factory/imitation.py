"""imitation.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import imitation
    """
import os
from typing import Any, Dict

import pytorch_lightning as pl

from logic.src.configs import Config
from logic.src.models.policies.alns import VectorizedALNS
from logic.src.models.policies.hgs import VectorizedHGS
from logic.src.models.policies.hgs_alns import VectorizedHGSALNS
from logic.src.models.policies.random_local_search import (
    RandomLocalSearchPolicy,
)
from logic.src.utils.configs.config_loader import load_yaml_config
from logic.src.utils.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)


def _get_expert_policy(expert_name: str, env_name: str, cfg: Config) -> Any:
    """get expert policy.

    Args:
    expert_name (str): Description of expert_name.
    env_name (str): Description of env_name.
    cfg (Config): Description of cfg.

    Returns:
        Any: Description of return value.
    """
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
    """create imitation.

    Args:
    cfg (Config): Description of cfg.
    policy (Any): Description of policy.
    env (Any): Description of env.
    kw (Dict[str, Any]): Description of kw.

    Returns:
        Any: Description of return value.
    """
    from logic.src.pipeline.rl.core.imitation import ImitationLearning

    expert_policy = _get_expert_policy(cfg.rl.imitation.mode, cfg.env.name, cfg)
    return ImitationLearning(expert_policy=expert_policy, expert_name=cfg.rl.imitation.mode, **kw)


def _create_adaptive_imitation(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    """create adaptive imitation.

    Args:
    cfg (Config): Description of cfg.
    policy (Any): Description of policy.
    env (Any): Description of env.
    kw (Dict[str, Any]): Description of kw.

    Returns:
        Any: Description of return value.
    """
    from logic.src.pipeline.rl.core.adaptive_imitation import AdaptiveImitation

    expert_policy = _get_expert_policy(cfg.rl.imitation.mode, cfg.env.name, cfg)
    return AdaptiveImitation(expert_policy=expert_policy, **kw)


def _create_critic_helper(policy, cfg: Config) -> Any:
    """create critic helper.

    Args:
    policy (Any): Description of policy.
    cfg (Config): Description of cfg.

    Returns:
        Any: Description of return value.
    """
    from logic.src.models.critic_network.policy import create_critic_from_actor

    return create_critic_from_actor(
        policy,
        env_name=cfg.env.name,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_encode_layers,
        n_heads=cfg.model.n_heads,
    )
