from dataclasses import dataclass, field
from typing import Dict, List, Optional


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

    # PPO specific
    ppo_epochs: int = 10
    eps_clip: float = 0.2
    value_loss_weight: float = 0.5
    mini_batch_size: float = 0.25

    # SAPO specific
    sapo_tau_pos: float = 0.1
    sapo_tau_neg: float = 1.0

    # DR-GRPO specific
    dr_grpo_group_size: int = 8
    dr_grpo_epsilon: float = 0.2

    # POMO / Augmentation specific
    num_augment: int = 1
    num_starts: Optional[int] = None
    augment_fn: str = "dihedral8"

    # SymNCO specific
    symnco_alpha: float = 0.2
    symnco_beta: float = 1.0

    # Imitation
    imitation_mode: str = "hgs"  # hgs, alns, random_ls, 2opt
    imitation_weight: float = 0.0
    imitation_decay: float = 1.0
    imitation_threshold: float = 0.05
    reannealing_threshold: float = 0.05
    reannealing_patience: int = 5
    random_ls_iterations: int = 100
    random_ls_op_probs: Optional[Dict[str, float]] = None

    # GDPO specific
    gdpo_objective_keys: List[str] = field(default_factory=lambda: ["reward_prize", "reward_cost"])
    gdpo_objective_weights: Optional[List[float]] = None
    gdpo_conditional_key: Optional[str] = None
    gdpo_renormalize: bool = True

    # Other algorithm parameters
    gamma: float = 0.99
    gspo_epsilon: float = 0.2
    gspo_epochs: int = 3
    dr_grpo_epochs: int = 3
    exp_beta: float = 0.8
    bl_alpha: float = 0.05
    imitation_decay_step: int = 1
