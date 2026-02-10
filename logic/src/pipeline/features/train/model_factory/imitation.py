"""Imitation learning model factory.

Creates imitation learning modules from configuration.
"""

from typing import Any, Dict

import pytorch_lightning as pl

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

    return ImitationLearning(
        policy_config=cfg.rl.imitation.policy_config,
        env_name=cfg.env.name,
        loss_fn=cfg.rl.imitation.loss_fn,
        **kw,
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

    return AdaptiveImitation(
        policy_config=cfg.rl.adaptive_imitation.policy_config,
        env_name=cfg.env.name,
        il_weight=cfg.rl.adaptive_imitation.il_weight,
        il_decay=cfg.rl.adaptive_imitation.il_decay,
        patience=cfg.rl.adaptive_imitation.patience,
        threshold=cfg.rl.adaptive_imitation.threshold,
        decay_step=cfg.rl.adaptive_imitation.decay_step,
        epsilon=cfg.rl.adaptive_imitation.epsilon,
        loss_fn=cfg.rl.adaptive_imitation.loss_fn,
        **kw,
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
        embed_dim=cfg.model.embed_dim,  # type: ignore[attr-defined]
        hidden_dim=cfg.model.hidden_dim,  # type: ignore[attr-defined]
        n_layers=cfg.model.n_encode_layers,  # type: ignore[attr-defined]
        n_heads=cfg.model.n_heads,  # type: ignore[attr-defined]
    )
