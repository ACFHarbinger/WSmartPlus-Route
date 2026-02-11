"""Imitation learning model factory.

Creates imitation learning modules from configuration.
"""

from typing import Any, Dict

import pytorch_lightning as pl
from hydra.utils import instantiate

from logic.src.configs import Config


def _create_imitation(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    """Create imitation learning module.

    Args:
        cfg: Full training configuration.
        policy: Student policy to train.
        env: Environment instance.
        kw: Additional keyword arguments for the module.

    Returns:
        Configured ImitationLearning module.
    """
    from logic.src.pipeline.rl.core.imitation import ImitationLearning

    # Validate policy config is provided
    if cfg.rl.imitation.policy_config is None:
        raise ValueError("imitation.policy_config must be provided for ImitationLearning")

    # Instantiate policy config if it has _target_ (Hydra config) or is a dict with _target_ key
    policy_config = cfg.rl.imitation.policy_config
    if hasattr(policy_config, "_target_") or (isinstance(policy_config, dict) and "_target_" in policy_config):
        policy_config = instantiate(policy_config)

    return ImitationLearning(
        # ImitationLearning specific
        policy_config=policy_config,
        env_name=cfg.env.name,
        loss_fn=cfg.rl.imitation.loss_fn,
        # RL4COLitModule parameters
        env=kw["env"],
        policy=kw["policy"],
        baseline=kw.get("baseline", "rollout"),
        optimizer=kw.get("optimizer", "adam"),
        optimizer_kwargs=kw.get("optimizer_kwargs", {}),
        lr_scheduler=kw.get("lr_scheduler"),
        lr_scheduler_kwargs=kw.get("lr_scheduler_kwargs", {}),
        train_data_size=kw.get("train_data_size", 100000),
        val_data_size=kw.get("val_data_size", 10000),
        val_dataset_path=kw.get("val_dataset_path"),
        train_dataset_path=kw.get("train_dataset_path"),
        batch_size=kw.get("batch_size", 256),
        eval_batch_size=kw.get("eval_batch_size", 256),
        num_workers=kw.get("num_workers", 4),
        persistent_workers=kw.get("persistent_workers", True),
        pin_memory=kw.get("pin_memory", False),
        must_go_selector=kw.get("must_go_selector"),
    )


def _create_adaptive_imitation(cfg: Config, policy, env, kw: Dict[str, Any]) -> pl.LightningModule:
    """Create adaptive imitation learning module.

    Args:
        cfg: Full training configuration.
        policy: Student policy to train.
        env: Environment instance.
        kw: Additional keyword arguments for the module.

    Returns:
        Configured AdaptiveImitation module.
    """
    from logic.src.pipeline.rl.core.adaptive_imitation import AdaptiveImitation

    # Validate policy config is provided
    if cfg.rl.adaptive_imitation.policy_config is None:
        raise ValueError("adaptive_imitation.policy_config must be provided for AdaptiveImitation")

    # Instantiate policy config if it has _target_ (Hydra config) or is a dict with _target_ key
    policy_config = cfg.rl.adaptive_imitation.policy_config
    if hasattr(policy_config, "_target_") or (isinstance(policy_config, dict) and "_target_" in policy_config):
        policy_config = instantiate(policy_config)

    return AdaptiveImitation(
        # AdaptiveImitation specific
        policy_config=policy_config,
        env_name=cfg.env.name,
        il_weight=cfg.rl.adaptive_imitation.il_weight,
        il_decay=cfg.rl.adaptive_imitation.il_decay,
        patience=cfg.rl.adaptive_imitation.patience,
        threshold=cfg.rl.adaptive_imitation.threshold,
        decay_step=cfg.rl.adaptive_imitation.decay_step,
        epsilon=cfg.rl.adaptive_imitation.epsilon,
        loss_fn=cfg.rl.adaptive_imitation.loss_fn,
        # REINFORCE parameters
        entropy_weight=kw.get("entropy_weight", 0.0),
        max_grad_norm=kw.get("max_grad_norm", 1.0),
        lr_critic=kw.get("lr_critic", 1e-4),
        # RL4COLitModule parameters
        env=kw["env"],
        policy=kw["policy"],
        baseline=kw.get("baseline", "rollout"),
        optimizer=kw.get("optimizer", "adam"),
        optimizer_kwargs=kw.get("optimizer_kwargs", {}),
        lr_scheduler=kw.get("lr_scheduler"),
        lr_scheduler_kwargs=kw.get("lr_scheduler_kwargs", {}),
        train_data_size=kw.get("train_data_size", 100000),
        val_data_size=kw.get("val_data_size", 10000),
        val_dataset_path=kw.get("val_dataset_path"),
        train_dataset_path=kw.get("train_dataset_path"),
        batch_size=kw.get("batch_size", 256),
        eval_batch_size=kw.get("eval_batch_size", 256),
        num_workers=kw.get("num_workers", 4),
        persistent_workers=kw.get("persistent_workers", True),
        pin_memory=kw.get("pin_memory", False),
        must_go_selector=kw.get("must_go_selector"),
    )


def _create_critic_helper(policy, cfg: Config) -> Any:
    """Create critic network from actor policy.

    Args:
        policy: Actor policy to create critic from.
        cfg: Training configuration.

    Returns:
        Initialized critic network.
    """
    from logic.src.models.critic_network.policy import create_critic_from_actor

    return create_critic_from_actor(
        policy,
        env_name=cfg.env.name,
        embed_dim=cfg.model.encoder.embed_dim,
        hidden_dim=cfg.model.encoder.hidden_dim,
        n_layers=cfg.model.encoder.n_layers,
        n_heads=cfg.model.encoder.n_heads,
    )
