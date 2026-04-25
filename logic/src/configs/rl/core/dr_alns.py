"""DR-ALNS specific configuration.

Attributes:
    DRALNSConfig: Configuration for DR-ALNS algorithm.

Example:
    dr_alns_config = DRALNSConfig(
        max_iterations=100,
        n_destroy_ops=3,
        n_repair_ops=2,
        n_steps_per_epoch=2048,
        batch_size=64,
        n_epochs=10,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )
"""

from dataclasses import dataclass


@dataclass
class DRALNSConfig:
    """DR-ALNS specific configuration.

    Attributes:
        max_iterations: Maximum number of iterations.
        n_destroy_ops: Number of destroy operations to use.
        n_repair_ops: Number of repair operations to use.
        n_steps_per_epoch: Number of steps per epoch.
        batch_size: Batch size for training.
        n_epochs: Number of epochs to train.
        lr: Learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_epsilon: Clip epsilon for PPO.
        value_loss_coef: Coefficient for value loss.
        entropy_coef: Coefficient for entropy.
        max_grad_norm: Maximum gradient norm.
    """

    max_iterations: int = 100
    n_destroy_ops: int = 3
    n_repair_ops: int = 2
    n_steps_per_epoch: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
