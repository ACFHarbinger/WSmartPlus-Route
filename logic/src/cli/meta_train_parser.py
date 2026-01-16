"""
Meta-Reinforcement Learning training argument parser.
"""

from logic.src.cli.train_parser import add_train_args


def add_mrl_train_args(parser):
    """
    Adds arguments for Meta-Reinforcement Learning (inherits from train_args).

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added MRL arguments.
    """
    parser = add_train_args(parser)

    # MRL specific args
    parser.add_argument(
        "--mrl_method",
        type=str,
        default="cb",
        choices=["tdl", "rwa", "cb", "morl", "hrl"],
        help="Method to use for Meta-Reinforcement Learning",
    )
    parser.add_argument(
        "--mrl_history",
        type=int,
        default=10,
        help="Number of previous days/epochs to take into account during Meta-Reinforcement Learning",
    )
    parser.add_argument(
        "--mrl_range",
        type=float,
        nargs="+",
        default=[0.01, 5.0],
        help="Maximum and minimum values for Meta-Reinforcement Learning with dynamic hyperparameters",
    )
    parser.add_argument(
        "--mrl_exploration_factor",
        type=float,
        default=2.0,
        help="Factor that controls the exploration vs. exploitation balance",
    )
    parser.add_argument(
        "--mrl_lr",
        type=float,
        default=1e-3,
        help="Set the learning rate for Meta-Reinforcement Learning",
    )
    parser.add_argument(
        "--mrl_embedding_dim",
        type=int,
        default=128,
        help="Dimension of input embedding for Reward Weight Adjustment model",
    )
    parser.add_argument("--mrl_step", type=int, default=100, help="Update every mrl_step steps")
    parser.add_argument(
        "--mrl_batch_size",
        type=int,
        default=256,
        help="Batch size to use for Meta-Reinforcement Learning",
    )
    parser.add_argument(
        "--hrl_threshold",
        type=float,
        default=0.9,
        help="Set the critical threshold for Hierarchical Reinforcement Learning PPO",
    )
    parser.add_argument(
        "--hrl_epochs",
        type=int,
        default=4,
        help="Number of epochs to use for Hierarchical Reinforcement Learning PPO",
    )
    parser.add_argument(
        "--hrl_clip_eps",
        type=float,
        default=0.2,
        help="Set the clip epsilon for Hierarchical Reinforcement Learning PPO",
    )
    parser.add_argument(
        "--shared_encoder",
        action="store_true",
        default=True,
        help="Set to share the encoder between worker and manager in HRL",
    )
    parser.add_argument(
        "--global_input_dim",
        type=int,
        default=2,
        help="Dimension of global input for HRL Manager",
    )
    parser.add_argument("--gat_hidden", type=int, default=128, help="Hidden dimension for GAT Manager")
    parser.add_argument(
        "--lstm_hidden",
        type=int,
        default=64,
        help="Hidden dimension for LSTM in GAT Manager",
    )
    parser.add_argument(
        "--gate_prob_threshold",
        type=float,
        default=0.5,
        help="Threshold for routing decision gate",
    )
    parser.add_argument(
        "--tdl_lr_decay",
        type=float,
        default=1.0,
        help="Learning rate decay for Temporal Difference Learning",
    )
    parser.add_argument(
        "--cb_exploration_method",
        type=str,
        default="ucb",
        help="Method for exploration in Contextual Bandits: 'ucb'|'thompson_sampling'|'epsilon_greedy'",
    )
    parser.add_argument(
        "--cb_num_configs",
        type=int,
        default=10,
        help="Number of weight configurations to generate in Contextual Bandits",
    )
    parser.add_argument(
        "--cb_context_features",
        type=str,
        nargs="+",
        default=["waste", "overflow", "length", "visited_ratio", "day"],
        help="Features for Contextual Bandits",
    )
    parser.add_argument(
        "--cb_features_aggregation",
        default="avg",
        help="Context features aggregation function in Contextual Bandits: 'sum'|'avg'|'max'",
    )
    parser.add_argument(
        "--cb_epsilon_decay",
        type=float,
        default=0.995,
        help="Decay factor for epsilon (=exploration_factor in 'epsilon_greedy')",
    )
    parser.add_argument(
        "--cb_min_epsilon",
        type=float,
        default=0.01,
        help="Minimum value for epsilon (=exploration_factor in 'epsilon_greedy')",
    )
    parser.add_argument(
        "--morl_objectives",
        type=str,
        nargs="+",
        default=["waste_efficiency", "overflow_rate"],
        help="Objectives for Multi-Objective RL",
    )
    parser.add_argument(
        "--morl_adaptation_rate",
        type=float,
        default=0.1,
        help="Adaptation rate in Multi-Objective RL",
    )
    parser.add_argument(
        "--rwa_model",
        type=str,
        default="rnn",
        choices=["rnn"],
        help="Neural network to use for Reward Weight Adjustment",
    )
    parser.add_argument(
        "--rwa_optimizer",
        type=str,
        default="rmsprop",
        help="Optimizer: 'adamax'|'adam'|'adamw'|'radam'|'nadam'|'rmsprop'",
    )
    parser.add_argument(
        "--rwa_update_step",
        type=int,
        default=100,
        help="Update Reward Weight Adjustment weights every rwa_update_step steps",
    )

    # HRL PID and Reward Shaping
    parser.add_argument(
        "--hrl_pid_target",
        type=float,
        default=0.05,
        help="Target overflow rate for PID control",
    )
    parser.add_argument(
        "--hrl_kp",
        type=float,
        default=50.0,
        help="Kp factor for HRL PID overflow control",
    )
    parser.add_argument(
        "--hrl_ki",
        type=float,
        default=5.0,
        help="Ki factor for HRL PID overflow control",
    )
    parser.add_argument(
        "--hrl_kd",
        type=float,
        default=0.0,
        help="Kd factor for HRL PID overflow control",
    )
    parser.add_argument(
        "--hrl_lambda_overflow_initial",
        type=float,
        default=1000.0,
        help="Initial lambda weight for overflows",
    )
    parser.add_argument(
        "--hrl_lambda_overflow_min",
        type=float,
        default=100.0,
        help="Minimum lambda weight for overflows",
    )
    parser.add_argument(
        "--hrl_lambda_overflow_max",
        type=float,
        default=2000.0,
        help="Maximum lambda weight for overflows",
    )
    parser.add_argument(
        "--hrl_lambda_waste",
        type=float,
        default=300.0,
        help="Reward weight for collected waste",
    )
    parser.add_argument(
        "--hrl_lambda_cost",
        type=float,
        default=0.1,
        help="Penalty weight for route cost/distance",
    )
    parser.add_argument(
        "--hrl_lambda_pruning",
        type=float,
        default=5.0,
        help="Penalty weight for masking too many nodes",
    )
    parser.add_argument(
        "--hrl_reward_scale",
        type=float,
        default=0.0001,
        help="Final scaling factor for HRL rewards",
    )

    # HRL Training Hyperparameters
    parser.add_argument(
        "--hrl_gamma",
        type=float,
        default=0.95,
        help="Discount factor for manager PPO update",
    )
    parser.add_argument(
        "--hrl_lambda_mask_aux",
        type=float,
        default=50.0,
        help="Weight for mask auxiliary loss",
    )
    parser.add_argument(
        "--hrl_entropy_coef",
        type=float,
        default=0.2,
        help="Entropy coefficient for manager PPO update",
    )

    return parser
