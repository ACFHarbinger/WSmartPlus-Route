"""RL algorithm configuration."""

from dataclasses import dataclass, field

from .adaptive_imitation import AdaptiveImitationConfig
from .gdpo import GDPOConfig
from .grpo import GRPOConfig
from .imitation import ImitationConfig
from .pomo import POMOConfig
from .ppo import PPOConfig
from .sapo import SAPOConfig
from .symnco import SymNCOConfig


@dataclass
class RLConfig:
    """RL algorithm configuration.

    Attributes:
        algorithm: RL algorithm name ('reinforce', 'ppo', 'sapo', etc.).
        baseline: Baseline type ('rollout', 'critic', 'pomo', 'warmup').
        entropy_weight: Weight for entropy regularization.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    algorithm: str = "reinforce"
    baseline: str = "rollout"
    bl_warmup_epochs: int = 0
    entropy_weight: float = 0.0
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    exp_beta: float = 0.8
    bl_alpha: float = 0.05

    # Algorithm specific sub-configs
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sapo: SAPOConfig = field(default_factory=SAPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    pomo: POMOConfig = field(default_factory=POMOConfig)
    symnco: SymNCOConfig = field(default_factory=SymNCOConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    gdpo: GDPOConfig = field(default_factory=GDPOConfig)
    adaptive_imitation: AdaptiveImitationConfig = field(default_factory=AdaptiveImitationConfig)
