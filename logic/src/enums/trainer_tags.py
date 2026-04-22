"""
Trainer enum for WSmart-Route.

Attributes:
    TrainerTag: Enum for trainer tags

Example:
    >>> from logic.src.enums import TrainerTag
    >>> TrainerTag.REINFORCEMENT_LEARNING
    <TrainerTag.REINFORCEMENT_LEARNING: 1>
"""

from enum import Enum, auto


class TrainerTag(Enum):
    """
    Trainer tags for WSmart-Route.

    Attributes:
        REINFORCEMENT_LEARNING: Reinforcement Learning
        SUPERVISED_LEARNING: Supervised Learning
        IMITATION_LEARNING: Imitation Learning
        ACTIVE_SEARCH: Test-time instance-specific training
        POLICY_GRADIENT: Policy Gradient (REINFORCE, PPO)
        VALUE_BASED: Value-Based (DQN)
        ACTOR_CRITIC: Actor-Critic (A2C, PPO)
        DISTRIBUTED_DATA_PARALLEL: Supports multi-GPU (DDP)
        REQUIRES_BASE_MODEL: Requires a pre-trained model
    """

    # Learning Paradigm
    REINFORCEMENT_LEARNING = auto()
    SUPERVISED_LEARNING = auto()
    IMITATION_LEARNING = auto()
    ACTIVE_SEARCH = auto()  # Test-time instance-specific training

    # RL Family
    POLICY_GRADIENT = auto()  # REINFORCE, PPO
    VALUE_BASED = auto()  # DQN
    ACTOR_CRITIC = auto()  # A2C, PPO

    # Hardware Context
    DISTRIBUTED_DATA_PARALLEL = auto()  # Supports multi-GPU (DDP)
    REQUIRES_BASE_MODEL = auto()  # e.g., Active Search needs a pre-trained model
