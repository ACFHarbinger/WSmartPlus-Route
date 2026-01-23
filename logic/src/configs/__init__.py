"""
Configuration dataclasses for WSmart-Route.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnvConfig:
    """Environment configuration.

    Attributes:
        name: Name of the environment (e.g., 'vrpp', 'wcvrp').
        num_loc: Number of locations (including depot).
        min_loc: Minimum coordinate value.
        max_loc: Maximum coordinate value.
        capacity: Vehicle capacity (optional).
    """

    name: str = "vrpp"
    num_loc: int = 50
    min_loc: float = 0.0
    max_loc: float = 1.0
    capacity: Optional[float] = None
    overflow_penalty: float = 1.0
    collection_reward: float = 1.0
    cost_weight: float = 1.0
    prize_weight: float = 1.0
    # NEW FIELDS:
    area: str = "riomaior"
    waste_type: str = "plastic"
    focus_graph: Optional[str] = None
    focus_size: int = 0
    eval_focus_size: int = 0
    distance_method: str = "ogd"
    dm_filepath: Optional[str] = None
    waste_filepath: Optional[str] = None
    vertex_method: str = "mmn"
    edge_threshold: float = 0.0
    edge_method: Optional[str] = None
    # Data distribution and generation
    data_distribution: Optional[str] = None
    min_fill: float = 0.0
    max_fill: float = 1.0
    fill_distribution: str = "uniform"


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        name: Name of the model architecture (e.g., 'am', 'deep_decoder').
        embed_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        encoder_type: Type of encoder ('gat', 'gcn', etc.).
    """

    name: str = "am"
    embed_dim: int = 128
    hidden_dim: int = 512
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_heads: int = 8
    encoder_type: str = "gat"
    # NEW FIELDS:
    temporal_horizon: int = 0
    tanh_clipping: float = 10.0
    normalization: str = "instance"
    activation: str = "gelu"
    dropout: float = 0.1
    mask_inner: bool = True
    mask_logits: bool = True
    mask_graph: bool = False
    spatial_bias: bool = False
    connection_type: str = "residual"
    # Hyper-parameters for specialized layers
    num_encoder_sublayers: Optional[int] = None
    num_predictor_layers: Optional[int] = None
    learn_affine: bool = True
    track_stats: bool = False
    epsilon_alpha: float = 1e-5
    momentum_beta: float = 0.1
    lrnorm_k: Optional[float] = None
    gnorm_groups: int = 4
    activation_param: float = 1.0
    activation_threshold: Optional[float] = None
    activation_replacement: Optional[float] = None
    activation_num_parameters: int = 3
    activation_uniform_range: List[float] = field(default_factory=lambda: [0.125, 0.333])
    aggregation_graph: str = "avg"
    aggregation_node: str = "sum"
    spatial_bias_scale: float = 1.0
    hyper_expansion: int = 4


@dataclass
class TrainConfig:
    """Training configuration.

    Attributes:
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        train_data_size: Number of training samples per epoch.
        val_data_size: Number of validation samples.
        val_dataset: Path to pre-generated validation dataset.
        num_workers: Number of data loading workers.
    """

    n_epochs: int = 100
    batch_size: int = 256
    train_data_size: int = 100000
    val_data_size: int = 10000
    val_dataset: Optional[str] = None
    num_workers: int = 4
    precision: str = "16-mixed"  # "16-mixed", "bf16-mixed", "32-true"
    # NEW FIELDS:
    train_time: bool = False
    eval_time_days: int = 1
    accumulation_steps: int = 1
    enable_scaler: bool = False
    checkpoint_epochs: int = 1
    shrink_size: Optional[int] = None
    post_processing_epochs: int = 0
    lr_post_processing: float = 0.001
    efficiency_weight: float = 0.8
    overflow_weight: float = 0.2
    log_step: int = 50
    # Process control
    epoch_start: int = 0
    eval_only: bool = False
    checkpoint_encoder: bool = False
    load_path: Optional[str] = None
    resume: Optional[str] = None
    eval_batch_size: int = 256


@dataclass
class OptimConfig:
    """Optimizer configuration.

    Attributes:
        optimizer: Name of the optimizer ('adam', 'sgd', etc.).
        lr: Learning rate.
        weight_decay: Weight decay factor.
        lr_scheduler: Name of the learning rate scheduler.
        lr_scheduler_kwargs: Keyword arguments for the LR scheduler.
    """

    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler: Optional[str] = None
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Learning rate details
    lr_critic: float = 1e-4
    lr_decay: float = 1.0
    lr_min_value: float = 0.0
    lr_min_decay: float = 1e-8


@dataclass
class RLConfig:
    """RL algorithm configuration.

    Attributes:
        algorithm: RL algorithm name ('reinforce', 'ppo', 'sapo', etc.).
        baseline: Baseline type ('rollout', 'critic', 'pomo', 'warmup').
        entropy_weight: Weight for entropy regularization.
        max_grad_norm: Maximum gradient norm for clipping.
        ppo_epochs: Number of PPO inner epochs.
        eps_clip: PPO clipping epsilon.
        value_loss_weight: Weight for value loss in PPO/Actor-Critic.
        sapo_tau_pos: SAPO positive threshold.
        sapo_tau_neg: SAPO negative threshold.
        dr_grpo_group_size: DR-GRPO group size.
        dr_grpo_epsilon: DR-GRPO divergence epsilon.
        use_meta: Whether to use meta-learning wrapper.
        meta_lr: Learning rate for meta-optimizer.
        meta_hidden_dim: Hidden dimension for meta-network.
        meta_history_length: History length for meta-learning.
        num_augment: Number of augmentations for POMO/SymNCO.
        num_starts: Number of start nodes for POMO/SymNCO.
        augment_fn: Augmentation function name.
        symnco_alpha: SymNCO alpha parameter.
        symnco_beta: SymNCO beta parameter.
        expert: Expert name for imitation learning ('hgs', 'alns', 'random_ls').
        random_ls_iterations: Iterations for random local search expert.
        random_ls_op_probs: Probabilities for local search operators.
        gdpo_objective_keys: List of reward keys for GDPO.
        gdpo_objective_weights: Weights for GDPO objectives.
        gdpo_conditional_key: Conditional gating key for GDPO.
        gdpo_renormalize: Whether to re-normalize aggregated advantage in GDPO.
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

    # Meta-RL
    use_meta: bool = False
    meta_strategy: str = "rnn"  # rnn|bandit|morl|tdl|hypernet|hrl
    meta_lr: float = 1e-3
    meta_hidden_dim: int = 64
    meta_history_length: int = 10
    mrl_exploration_factor: float = 2.0
    mrl_range: List[float] = field(default_factory=lambda: [0.01, 5.0])
    mrl_batch_size: int = 256
    mrl_step: int = 10

    # HRL
    hrl_threshold: float = 0.9
    hrl_epochs: int = 4
    hrl_clip_eps: float = 0.2
    hrl_pid_target: float = 0.0003
    hrl_lambda_waste: float = 300.0
    hrl_lambda_cost: float = 0.5
    hrl_lambda_overflow_initial: float = 2000.0
    hrl_lambda_overflow_min: float = 100.0
    hrl_lambda_overflow_max: float = 5000.0
    hrl_lambda_pruning: float = 0.5
    hrl_lambda_mask_aux: float = 5.0
    hrl_entropy_coef: float = 0.01
    shared_encoder: bool = True
    gat_hidden_dim: int = 128
    lstm_hidden_dim: int = 64
    gate_prob_threshold: float = 0.5
    lr_critic_value: float = 1e-4

    # Contextual Bandits
    cb_exploration_method: str = "ucb"
    cb_num_configs: int = 10
    cb_epsilon_decay: float = 0.995
    cb_min_epsilon: float = 0.01

    # GDPO specific
    gdpo_objective_keys: List[str] = field(default_factory=lambda: ["reward_prize", "reward_cost"])
    gdpo_objective_weights: Optional[List[float]] = None
    gdpo_conditional_key: Optional[str] = None
    gdpo_renormalize: bool = True
    # Missing from parsers
    gamma: float = 0.99
    gspo_epsilon: float = 0.2
    gspo_epochs: int = 3
    dr_grpo_epochs: int = 3
    exp_beta: float = 0.8
    bl_alpha: float = 0.05
    cb_context_features: List[str] = field(
        default_factory=lambda: ["waste", "overflow", "length", "visited_ratio", "day"]
    )
    cb_features_aggregation: str = "avg"
    morl_objectives: List[str] = field(default_factory=lambda: ["waste_efficiency", "overflow_rate"])
    morl_adaptation_rate: float = 0.1
    imitation_decay_step: int = 1


@dataclass
class HPOConfig:
    """Hyperparameter optimization configuration.

    Attributes:
        method: HPO method ('dehbo', 'rs', 'gs', 'bo').
        metric: Optimization metric ('reward', 'cost').
        n_trials: Number of HPO trials.
        n_epochs_per_trial: Training epochs per trial.
        num_workers: Number of parallel workers for HPO.
        search_space: Dictionary defining the search space.
    """

    method: str = "dehbo"  # dehbo, rs, gs, bo
    metric: str = "reward"
    n_trials: int = 20
    n_epochs_per_trial: int = 10
    num_workers: int = 4
    search_space: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "rl.entropy_weight": [0.0, 0.1],
            "optim.lr": [1e-5, 1e-3],
        }
    )
    # NEW FIELDS:
    hop_range: List[float] = field(default_factory=lambda: [0.0, 2.0])
    fevals: int = 100
    timeout: Optional[int] = None
    n_startup_trials: int = 5
    n_warmup_steps: int = 3
    min_fidelity: int = 1
    max_fidelity: int = 10
    # Ray Tune and DEA specifics
    interval_steps: int = 1
    eta: float = 10.0
    indpb: float = 0.2
    tournsize: int = 3
    cxpb: float = 0.7
    mutpb: float = 0.2
    n_pop: int = 20
    n_gen: int = 10
    cpu_cores: int = 1
    verbose: int = 2
    train_best: bool = True
    local_mode: bool = False
    num_samples: int = 20
    max_tres: int = 14
    reduction_factor: int = 3
    max_failures: int = 3
    grid: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])
    max_conc: int = 4


@dataclass
class Config:
    """Root configuration.

    Attributes:
        env: Environment configuration.
        model: Model configuration.
        train: Training configuration.
        optim: Optimizer configuration.
        rl: RL algorithm configuration.
        hpo: HPO configuration.
        seed: Random seed.
        device: Device to use ('cpu', 'cuda').
        experiment_name: Optional name for the experiment.
    """

    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    seed: int = 42
    device: str = "cuda"
    experiment_name: Optional[str] = None
    # NEW FIELDS:
    wandb_mode: str = "offline"
    no_tensorboard: bool = False
    no_progress_bar: bool = False
    output_dir: str = "assets/model_weights"
    log_dir: str = "logs"
    run_name: Optional[str] = None


__all__ = [
    "EnvConfig",
    "ModelConfig",
    "TrainConfig",
    "OptimConfig",
    "RLConfig",
    "HPOConfig",
    "Config",
]


__all__ = [
    "EnvConfig",
    "ModelConfig",
    "TrainConfig",
    "OptimConfig",
    "RLConfig",
    "HPOConfig",
    "Config",
]
