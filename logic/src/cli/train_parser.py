"""
Training-related argument parsers.
"""
import os
import re
import time
import argparse
from multiprocessing import cpu_count

from logic.src.cli.base_parser import LowercaseAction, StoreDictKeyPair
from logic.src.utils.functions import parse_softmax_temperature
from logic.src.utils.definitions import (
    MAP_DEPOTS, WASTE_TYPES, SUB_NET_ENCS, PRED_ENC_MODELS, ENC_DEC_MODELS
)

def add_train_args(parser):
    """
    Adds all arguments related to training to the given parser.

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added arguments.
    """
    # Data
    parser.add_argument("--problem", default="wcvrp", help="The problem to solve")
    parser.add_argument(
        "--graph_size", type=int, default=20, help="The size of the problem graph"
    )
    parser.add_argument(
        "--edge_threshold",
        default="0",
        type=str,
        help="How many of all possible edges to consider",
    )
    parser.add_argument(
        "--edge_method",
        type=str,
        default=None,
        help="Method for getting edges ('dist'|'knn')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of instances per batch during training",
    )
    parser.add_argument(
        "--epoch_size",
        type=int,
        default=128_000,
        help="Number of instances per epoch during training",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=0,
        help="Number of instances used for reporting validation performance (0 to deactivate validation)",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default=None,
        help="Dataset file to use for validation",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size to use during (baseline) evaluation",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default=None,
        help="Name of dataset to use for training",
    )

    # WSmart+ Route
    parser.add_argument(
        "--eval_time_days",
        type=int,
        default=1,
        help="Number of days to perform validation (if train_time=True)",
    )
    parser.add_argument(
        "--train_time",
        action="store_true",
        help="Set to train the model over multiple days on the same graphs (n_days=n_epochs)",
    )
    parser.add_argument(
        "--area", type=str, default="riomaior", help="County area of the bins locations"
    )
    parser.add_argument(
        "--waste_type",
        type=str,
        default="plastic",
        help="Type of waste bins selected for the optimization problem",
    )
    parser.add_argument(
        "--focus_graph",
        default=None,
        help="Path to the file with the coordinates of the graph to focus on",
    )
    parser.add_argument(
        "--focus_size",
        type=int,
        default=0,
        help="Number of focus graphs to include in the training data",
    )
    parser.add_argument(
        "--eval_focus_size",
        type=int,
        default=0,
        help="Number of focus graphs to include in the validation data",
    )
    parser.add_argument(
        "--distance_method",
        type=str,
        default="ogd",
        help="Method to compute distance matrix",
    )
    parser.add_argument(
        "--dm_filepath",
        type=str,
        default=None,
        help="Path to the file to read/write the distance matrix from/to",
    )
    parser.add_argument(
        "--waste_filepath",
        type=str,
        default=None,
        help="Path to the file to read the waste fill for each day from",
    )
    parser.add_argument(
        "--vertex_method",
        type=str,
        default="mmn",
        help="Method to transform vertex coordinates "
        "'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'",
    )

    # Model
    parser.add_argument("--model", default="am", help="Model: 'am'|'tam'|'ddam'")
    parser.add_argument(
        "--encoder", default="gat", help="Encoder: 'gat'|gac'|'tgc'|'ggac'|'gcn'|'mlp'"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Dimension of input embedding"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Dimension of hidden layers in Enc/Dec",
    )
    parser.add_argument(
        "--n_encode_layers",
        type=int,
        default=3,
        help="Number of layers in the encoder/critic network",
    )
    parser.add_argument(
        "--n_encode_sublayers",
        type=int,
        default=None,
        help="Number of sublayers in the encoder network",
    )
    parser.add_argument(
        "--n_predict_layers",
        type=int,
        default=None,
        help="Number of layers in the predictor network",
    )
    parser.add_argument(
        "--n_decode_layers",
        type=int,
        default=None,
        help="Number of layers in the decoder network",
    )
    parser.add_argument(
        "--temporal_horizon",
        type=int,
        default=0,
        help="Number of previous days/epochs to take into account in predictions",
    )
    parser.add_argument(
        "--tanh_clipping",
        type=float,
        default=10.0,
        help="Clip the parameters to within +- this value using tanh. "
        "Set to 0 to not perform any clipping.",
    )
    parser.add_argument(
        "--normalization",
        default="instance",
        help="Normalization type: 'instance'|'layer'|'batch'|'group'|'local_response'|None",
    )
    parser.add_argument(
        "--learn_affine",
        action="store_false",
        help="Disable learnable affine transformation during normalization",
    )
    parser.add_argument(
        "--track_stats",
        action="store_true",
        help="Track statistics during normalization",
    )
    parser.add_argument(
        "--epsilon_alpha",
        type=float,
        default=1e-05,
        help="Epsilon (or alpha multiplicative, for LocalResponseNorm) value",
    )
    parser.add_argument(
        "--momentum_beta",
        type=float,
        default=0.1,
        help="Momentum (or beta exponential, for LocalResponseNorm) value",
    )
    parser.add_argument(
        "--lrnorm_k",
        type=float,
        default=None,
        help="Additive factor for LocalResponseNorm",
    )
    parser.add_argument(
        "--gnorm_groups",
        type=int,
        default=4,
        help="Number of groups to separate channels into for GroupNorm",
    )
    parser.add_argument(
        "--activation",
        default="gelu",
        choices=[
            "gelu",
            "gelu_tanh",
            "tanh",
            "tanhshrink",
            "mish",
            "hardshrink",
            "hardtanh",
            "hardswish",
            "glu",
            "relu",
            "leakyrelu",
            "silu",
            "selu",
            "elu",
            "celu",
            "prelu",
            "rrelu",
            "sigmoid",
            "logsigmoid",
            "hardsigmoid",
            "threshold",
            "softplus",
            "softshrink",
            "softsign",
        ],
        help="Activation function: 'gelu'|'gelu_tanh'|'tanh'|'tanhshrink'|'mish'|'hardshrink'|'hardtanh'|'hardswish'|"
        "'glu'|'relu'|'leakyrelu'|'silu'|'selu'|'elu'|'celu'|'prelu'|'rrelu'|'sigmoid'|'logsigmoid'|'hardsigmoid'|"
        "'threshold'|'softplus'|'softshrink'|'softsign'",
    )
    parser.add_argument(
        "--af_param",
        type=float,
        default=1.0,
        help="Parameter for the activation function formulation",
    )
    parser.add_argument(
        "--af_threshold",
        type=float,
        default=None,
        help="Threshold value for the activation function",
    )
    parser.add_argument(
        "--af_replacement",
        type=float,
        default=None,
        help="Replacement value for the activation function (above/below threshold)",
    )
    parser.add_argument(
        "--af_nparams",
        type=int,
        default=3,
        help="Number of parameters a for the Parametric ReLU (PReLU) activation",
    )
    parser.add_argument(
        "--af_urange",
        type=float,
        nargs="+",
        default=[0.125, 1 / 3],
        help="Range for the uniform distribution of the Randomized Leaky ReLU (RReLU) activation",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate for the model"
    )
    parser.add_argument(
        "--aggregation_graph",
        default="avg",
        help="Graph embedding aggregation function: 'sum'|'avg'|'max'|None",
    )
    parser.add_argument(
        "--aggregation",
        default="sum",
        help="Node embedding aggregation function: 'sum'|'avg'|'max'",
    )
    parser.add_argument(
        "--n_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--mask_inner", action="store_false", help="Mask inner values during decoding"
    )
    parser.add_argument(
        "--mask_logits", action="store_false", help="Mask logits during decoding"
    )
    parser.add_argument(
        "--mask_graph",
        action="store_true",
        help="Mask next node selection (using edges) during decoding",
    )
    parser.add_argument(
        "--spatial_bias",
        action="store_true",
        help="Enable spatial bias in decoder attention",
    )
    parser.add_argument(
        "--spatial_bias_scale",
        type=float,
        default=1.0,
        help="Scaling factor for the spatial bias penalty",
    )
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=0.0,
        help="Weight for the entropy regularization",
    )
    parser.add_argument(
        "--imitation_weight",
        type=float,
        default=0.0,
        help="Initial weight for the imitation loss guidance",
    )
    parser.add_argument(
        "--imitation_threshold",
        type=float,
        default=0.05,
        help="Stop threshold for the imitation loss guidance",
    )
    parser.add_argument(
        "--imitation_decay",
        type=float,
        default=1.0,
        help="Decay factor for the imitation weight",
    )
    parser.add_argument(
        "--imitation_decay_step",
        type=int,
        default=1,
        help="Number of epochs after which to apply imitation decay",
    )
    parser.add_argument(
        "--reannealing_threshold",
        type=float,
        default=0.05,
        help="Performance gap threshold (percentage) to trigger reannealing",
    )
    parser.add_argument(
        "--reannealing_patience",
        type=int,
        default=5,
        help="Number of consecutive epochs of underperformance before reannealing",
    )
    parser.add_argument(
        "--imitation_mode",
        type=str,
        default="2opt",
        choices=["2opt", "hgs"],
        help="Method for imitation learning expert ('2opt'|'hgs')",
    )
    parser.add_argument(
        "--hgs_config_path",
        type=str,
        default=None,
        help="Path to HGS configuration file (YAML)",
    )
    parser.add_argument(
        "--two_opt_max_iter",
        type=int,
        default=0,
        help="Maximum number of iterations for 2-opt refinement in Look-Ahead update",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for future rewards"
    )
    parser.add_argument(
        "--rl_algorithm",
        type=str,
        default="reinforce",
        choices=["reinforce", "ppo", "sapo", "gspo", "dr_grpo"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--ppo_epochs", type=int, default=3, help="Number of epochs for PPO update"
    )
    parser.add_argument(
        "--ppo_eps_clip", type=float, default=0.2, help="PPO clip parameter"
    )
    parser.add_argument(
        "--ppo_mini_batch_size",
        type=int,
        default=32,
        help="Mini-batch size for PPO update",
    )
    parser.add_argument(
        "--sapo_tau_pos",
        type=float,
        default=0.1,
        help="Temperature for positive advantages in SAPO",
    )
    parser.add_argument(
        "--sapo_tau_neg",
        type=float,
        default=1.0,
        help="Temperature for negative advantages in SAPO",
    )
    parser.add_argument(
        "--gspo_epsilon", type=float, default=0.2, help="GSPO epsilon clip"
    )
    parser.add_argument(
        "--gspo_epochs", type=int, default=3, help="Number of GSPO epochs"
    )

    # DR-GRPO Args
    parser.add_argument(
        "--dr_grpo_group_size", type=int, default=8, help="Group size (G) for DR-GRPO"
    )
    parser.add_argument(
        "--dr_grpo_epsilon", type=float, default=0.2, help="DR-GRPO epsilon clip"
    )
    parser.add_argument(
        "--dr_grpo_epochs", type=int, default=3, help="Number of DR-GRPO epochs"
    )

    # Training
    parser.add_argument(
        "--n_epochs", type=int, default=25, help="The number of epochs to train"
    )
    parser.add_argument(
        "--epoch_start",
        type=int,
        default=0,
        help="Start at epoch # (relevant for learning rate decay)",
    )
    parser.add_argument(
        "--lr_model",
        type=float,
        default=1e-4,
        help="Set the learning rate for the actor network",
    )
    parser.add_argument(
        "--lr_critic_value",
        type=float,
        default=1e-4,
        help="Set the learning rate for the critic/value network",
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="Set this value to only evaluate model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum L2 norm for gradient clipping (0 to disable clipping)",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading (0 to deactivate multiprocessing)",
    )
    parser.add_argument(
        "--enable_scaler",
        action="store_true",
        help="Enables CUDA scaler for automatic mixed precision training",
    )
    parser.add_argument(
        "--exp_beta",
        type=float,
        default=0.8,
        help="Exponential moving average baseline decay",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline to use: 'rollout'|'critic'|'exponential'|'pomo'|None",
    )
    parser.add_argument(
        "--bl_alpha",
        type=float,
        default=0.05,
        help="Significance in the t-test for updating rollout baseline",
    )
    parser.add_argument(
        "--bl_warmup_epochs",
        type=int,
        default=-1,
        help="Number of epochs to warmup the baseline, "
        "None means 1 for rollout (exponential used for warmup phase), 0 otherwise. Can only be used with rollout baseline.",
    )
    parser.add_argument(
        "--checkpoint_encoder",
        action="store_true",
        help="Set to decrease memory usage by checkpointing encoder",
    )
    parser.add_argument(
        "--shrink_size",
        type=int,
        default=None,
        help="Shrink the batch size if at least this many instances in the batch"
        " are finished to save memory (default None means no shrinking)",
    )
    parser.add_argument(
        "--pomo_size",
        type=int,
        default=0,
        help="Number of starting nodes for POMO (Policy Optimization with Multiple Optima)",
    )
    parser.add_argument(
        "--data_distribution",
        type=str,
        default=None,
        help="Data distribution to use during training,"
        ' defaults and options depend on problem. "empty"|"const"|"unif"|"dist"|"gamma[1-4]|"emp""',
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps during training (effective batch_size = batch_size * accumulation_steps)",
    )
    parser.add_argument(
        "--load_path", help="Path to load model parameters and optimizer state from"
    )
    parser.add_argument("--resume", help="Resume from previous checkpoint file")
    parser.add_argument(
        "--post_processing_epochs",
        type=int,
        default=0,
        help="The number of epochs for post-processing",
    )
    parser.add_argument(
        "--lr_post_processing",
        type=float,
        default=0.001,
        help="Set the learning rate for post-processing",
    )
    parser.add_argument(
        "--efficiency_weight",
        type=float,
        default=0.8,
        help="Weight for the efficiency in post-processing",
    )
    parser.add_argument(
        "--overflow_weight",
        type=float,
        default=0.2,
        help="Weight for the bin overflows in post-processing",
    )

    # Optimizer and learning rate scheduler
    parser.add_argument(
        "--optimizer",
        type=str,
        default="rmsprop",
        help="Optimizer: 'adam'|'adamax'|'adamw'|'radam'|'nadam'|'sadam'|'adadelta'|'adagrad'|'rmsprop'|'rprop'|'lbfgs'|'asgd'|'sgd'",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="lambda",
        help="Learning rate scheduler: 'exp'|'step'|'mult'|'lambda'|'const'|'poly'|'multistep'|'cosan'|'linear'|'cosanwr'|'plateau'",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=1.0, help="Learning rate decay per epoch"
    )
    parser.add_argument(
        "--lr_min_value",
        type=float,
        default=0.0,
        help="Minimum learning rate for CosineAnnealingLR|CosineAnnealingWarmRestarts|ReduceLROnPlateau",
    )
    parser.add_argument(
        "--lr_min_decay",
        type=float,
        default=1e-8,
        help="Minimum decay applied to learning rate for LinearLR|ReduceLROnPlateau",
    )
    parser.add_argument(
        "--lrs_step_size",
        type=int,
        default=1,
        help="Period of learning rate decay for StepLR",
    )
    parser.add_argument(
        "--lrs_total_steps",
        type=int,
        default=5,
        help="Number of steps that the scheduler updates the lr for ConstantLR|LinearLR|PolynomialLR",
    )
    parser.add_argument(
        "--lrs_restart_steps",
        type=int,
        default=7,
        help="Number of steps until the first restart for CosineAnnealingWarmRestarts",
    )
    parser.add_argument(
        "--lrs_rfactor",
        type=int,
        default=2,
        help="A factor that, after a restart, increases the steps for the next restart for CosineAnnealingWarmRestarts.",
    )
    parser.add_argument(
        "--lrs_milestones",
        type=int,
        nargs="+",
        default=[7, 14, 21, 28],
        help="List of epoch indices (must be increasing) for MultiStepLR.",
    )
    parser.add_argument(
        "--lrs_mode",
        type=str,
        default="min",
        help="Scheduler mode for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--lrs_dfactor",
        type=float,
        default=0.1,
        help="A factor by which the learning rate will be decreased for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--lrs_patience",
        type=int,
        default=10,
        help="Number of epochs with no improvement after which the learning rate will be updated for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--lrs_thresh",
        type=float,
        default=1e-4,
        help="Threshold for measuring the new optimum, to only focus on significant changes for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--lrs_thresh_mode",
        type=str,
        default="rel",
        choices=["rel", "abs"],
        help="Scheduler threshold mode for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--lrs_cooldown",
        type=int,
        default=0,
        help="Number of epochs to wait before resuming normal operation after lr has been reduced for ReduceLROnPlateau",
    )

    # Cost function weights
    parser.add_argument(
        "--w_waste", type=float, default=None, help="Weight for the waste collected"
    )
    parser.add_argument(
        "--w_length",
        "--w_len",
        type=float,
        default=None,
        help="Weight for the route length",
    )
    parser.add_argument(
        "--w_overflows",
        "--w_over",
        type=float,
        default=None,
        help="Weight for the number of overflows",
    )
    parser.add_argument(
        "--w_lost",
        type=float,
        default=None,
        help="Weight for the amount of waste lost when bins overflow",
    )
    parser.add_argument(
        "--w_penalty",
        "--w_pen",
        type=float,
        default=None,
        help="Weight for the penalty",
    )
    parser.add_argument(
        "--w_prize", type=float, default=None, help="Weight for the prize"
    )

    # Output
    parser.add_argument(
        "--log_step", type=int, default=50, help="Log info every log_step steps"
    )
    parser.add_argument(
        "--log_dir",
        default="logs",
        help="Directory to write TensorBoard information to",
    )
    parser.add_argument("--run_name", default=None, help="Name to identify the run")
    parser.add_argument(
        "--output_dir",
        default="assets/model_weights",
        help="Directory to write output models to",
    )
    parser.add_argument(
        "--checkpoints_dir",
        default="model_weights",
        help="Directory to write checkpoints to",
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=1,
        help="Save checkpoint every n epochs, 0 to save no checkpoints",
    )
    parser.add_argument(
        "--wandb_mode",
        default="offline",
        help="Weights and biases mode: 'online'|'offline'|'disabled'|None",
    )
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable logging TensorBoard files",
    )
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )

    # Visualization
    parser.add_argument(
        "--visualize_step",
        type=int,
        default=50,
        help="Frequency of visualization (epochs). Default 0 (disabled).",
    )
    parser.add_argument(
        "--viz_modes",
        type=str,
        nargs="+",
        default=["trajectory", "distributions", "embeddings", "heatmaps", "loss", "logit_lens"],
        choices=[
            "trajectory",
            "distributions",
            "embeddings",
            "heatmaps",
            "logit_lens",
            "loss",
            "both",
        ],
        help="Visualization modes to run during training.",
    )

    # --- New Arguments for Hyper-Connections ---
    parser.add_argument(
        "--connection_type",
        type=str,
        default="residual",
        choices=["residual", "static_hyper", "dynamic_hyper"],
        help="Type of skip connection to use.",
    )

    parser.add_argument(
        "--hyper_expansion",
        type=int,
        default=4,
        help="Expansion rate (number of streams) for Hyper-Connections.",
    )

    return parser

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
    parser.add_argument(
        "--mrl_step", type=int, default=100, help="Update every mrl_step steps"
    )
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
    parser.add_argument(
        "--gat_hidden", type=int, default=128, help="Hidden dimension for GAT Manager"
    )
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

def add_hp_optim_args(parser):
    """
    Adds arguments for Hyper-Parameter Optimization (inherits from train_args).

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added HPO arguments.
    """
    parser = add_train_args(parser)

    # HPO specific args
    parser.add_argument(
        "--hop_method",
        type=str,
        default="dehbo",
        choices=["dea", "bo", "hbo", "rs", "gs", "dehbo"],
        help="Method to use for hyperparameter optimization",
    )
    parser.add_argument(
        "--hop_range",
        type=float,
        nargs="+",
        default=[0.0, 2.0],
        help="Maximum and minimum values for hyperparameter optimization",
    )
    parser.add_argument(
        "--hop_epochs",
        type=int,
        default=7,
        help="The number of epochs to optimize hyperparameters",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_loss",
        choices=[
            "loss",
            "val_loss",
            "mean_reward",
            "mae",
            "mse",
            "rmse",
            "episode_reward_mean",
            "kg/km",
            "overflows",
            "both",
        ],
        help="Metric to optimize",
    )

    # ===== Bayesian Optimization (BO) =====
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials for Optuna optimization",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout for Optuna optimization (in seconds)",
    )
    parser.add_argument(
        "--n_startup_trials",
        type=int,
        default=5,
        help="Number of trials to run before pruning starts",
    )
    parser.add_argument(
        "--n_warmup_steps",
        type=int,
        default=3,
        help="Number of epochs to wait before pruning can happen in each trial",
    )
    parser.add_argument(
        "--interval_steps",
        type=int,
        default=1,
        help="Pruning is evaluated every this many epochs",
    )

    # ===== Distributed Evolutionary Algorithm (DEA) =====
    parser.add_argument(
        "--eta",
        type=float,
        default=10.0,
        help="Controls the spread of the genetic mutations (higher = slower changes)",
    )
    parser.add_argument(
        "--indpb",
        type=float,
        default=0.2,
        help="Probability of mutating each gene of an individual in the population",
    )
    parser.add_argument(
        "--tournsize",
        type=int,
        default=3,
        help="Number of individuals to fight to be selected for reproduction",
    )
    parser.add_argument(
        "--cxpb",
        type=float,
        default=0.7,
        help="Probability of crossover between two parents (higher = faster convergence + less diversity)",
    )
    parser.add_argument(
        "--mutpb",
        type=float,
        default=0.2,
        help="Probability of an individual being mutated after crossover",
    )
    parser.add_argument(
        "--n_pop",
        type=int,
        default=20,
        help="Starting population for evolutionary algorithms",
    )
    parser.add_argument(
        "--n_gen", type=int, default=10, help="Number of generations to evolve"
    )

    # ===== Differential Evolutionary Hyperband Optimization (DEHBO) =====
    parser.add_argument(
        "--fevals", type=int, default=100, help="Number of function evaluations"
    )

    # Ray Tune framework hyperparameters
    parser.add_argument(
        "--cpu_cores",
        type=int,
        default=1,
        help="Number of CPU cores to use for hyperparameter optimization (0 uses all available cores)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Verbosity level for Hyperband and Random Search (0-3)",
    )
    parser.add_argument(
        "--train_best",
        action="store_true",
        default=True,
        help="Train final model with best hyperparameters",
    )
    parser.add_argument(
        "--local_mode",
        action="store_true",
        help="Run ray in local mode (for debugging)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of times to sample from the hyperparameter space",
    )

    # ===== Hyperband Optimization (HBO) - Ray Tune =====
    parser.add_argument(
        "--max_tres",
        type=int,
        default=14,
        help="Maximum resources (timesteps) per trial",
    )
    parser.add_argument(
        "--reduction_factor", type=int, default=3, help="Reduction factor for Hyperband"
    )

    # ===== Random Search (RS) - Ray Tune =====
    parser.add_argument(
        "--max_failures",
        type=int,
        default=3,
        help="Maximum trial failures before stopping",
    )

    # ===== Grid Search (GS) - Ray Tune =====
    parser.add_argument(
        "--grid",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
        help="Hyperparameter values to try in grid search",
    )
    parser.add_argument(
        "--max_conc",
        type=int,
        default=4,
        help="Maximum number of concurrent trials for Ray Tune",
    )
    return parser


def validate_train_args(args):
    """
    Validates and post-processes arguments for train, mrl_train, and hp_optim.

    Args:
        args (dict): The parsed arguments dictionary.

    Returns:
        dict: The validated and processed arguments dictionary.

    Raises:
        AssertionError: If any validation checks fail.
    """
    args = args.copy()
    assert (
        args["epoch_size"] % args["batch_size"] == 0
    ), "Epoch size must be integer multiple of batch size!"

    if args.get("bl_warmup_epochs", -1) < 0:
        args["bl_warmup_epochs"] = 1 if args.get("baseline") == "rollout" else 0

    assert (args["bl_warmup_epochs"] == 0) or (args.get("baseline") == "rollout")

    if args.get("baseline") == "pomo":
        assert (
            args.get("pomo_size", 0) > 0
        ), "pomo_size must be > 0 when using pomo baseline"

    if args.get("encoder") in SUB_NET_ENCS and args.get("n_encode_sublayers") is None:
        args["n_encode_sublayers"] = args["n_encode_layers"]

    assert (
        args.get("encoder") not in SUB_NET_ENCS or args.get("n_encode_sublayers", 0) > 0
    ), f"Must select a positive integer for 'n_encode_sublayers' arg for {args.get('encoder')} encoder"

    if args.get("model") in PRED_ENC_MODELS and args.get("n_predict_layers") is None:
        args["n_predict_layers"] = args["n_encode_layers"]

    assert (
        args.get("model") not in PRED_ENC_MODELS or args.get("n_predict_layers", 0) > 0
    ), f"Must select a positive integer for 'n_predict_layers' arg for {args.get('model')} model"

    if args.get("model") in ENC_DEC_MODELS and args.get("n_decode_layers") is None:
        args["n_decode_layers"] = args["n_encode_layers"]

    assert (
        args.get("model") not in ENC_DEC_MODELS or args.get("n_decode_layers", 0) > 0
    ), f"Must select a positive integer for 'n_decode_layers' arg for {args.get('model')} model"

    if args.get("run_name") is not None:
        args["run_name"] = "{}_{}".format(
            args["run_name"], time.strftime("%Y%m%dT%H%M%S")
        )
    else:
        args["run_name"] = "{}{}{}{}_{}".format(
            args.get("model", "model"),
            args.get("encoder", "enc"),
            (
                args.get("temporal_horizon", 0)
                if args.get("temporal_horizon", 0) > 0
                else ""
            ),
            (
                "_{}".format(args.get("data_distribution"))
                if args.get("data_distribution") is not None
                else ""
            ),
            time.strftime("%Y%m%dT%H%M%S"),
        )

    args["save_dir"] = os.path.join(
        args.get("checkpoints_dir", "model_weights"),
        "{}_{}".format(args.get("problem", "problem"), args.get("graph_size", "size")),
        args["run_name"],
    )

    args["final_dir"] = os.path.join(
        args.get("output_dir", "assets/model_weights"),
        "{}{}{}{}".format(
            args.get("problem", "problem"),
            args.get("graph_size", "size"),
            (
                "_{}".format(args.get("area"))
                if "area" in args and args["area"] is not None
                else ""
            ),
            (
                "_{}".format(args.get("waste_type"))
                if "waste_type" in args and args["waste_type"] is not None
                else ""
            ),
        ),
        args.get("data_distribution") or "",
        "{}{}{}".format(
            args.get("model", "model"),
            args.get("encoder", "enc"),
            (
                "_{}".format(args.get("mrl_method"))
                if "mrl_method" in args and args.get("mrl_method") is not None
                else ""
            ),
        ),
    )

    if "area" in args and args["area"] is not None:
        args["area"] = re.sub(r"[^a-zA-Z]", "", args["area"].lower())
        assert (
            args["area"] in MAP_DEPOTS.keys()
        ), "Unknown area {}, available areas: {}".format(
            args["area"], MAP_DEPOTS.keys()
        )

    if "waste_type" in args and args["waste_type"] is not None:
        assert "area" in args and args["area"] is not None
        args["waste_type"] = re.sub(r"[^a-zA-Z]", "", args["waste_type"].lower())
        assert (
            args["waste_type"] in WASTE_TYPES.keys() or args["waste_type"] is None
        ), "Unknown waste type {}, available waste types: {}".format(
            args["waste_type"], WASTE_TYPES.keys()
        )

    args["edge_threshold"] = (
        float(args["edge_threshold"])
        if "." in args["edge_threshold"]
        else int(args["edge_threshold"])
    )

    if "hop_method" in args:  # hp_optim specific
        assert (
            args.get("cpu_cores", 1) >= 0
        ), "Number of CPU cores must be non-negative integer"
        assert (
            args.get("cpu_cores", 1) <= cpu_count()
        ), "Number of CPU cores to use cannot exceed system specifications"
        if args.get("cpu_cores") == 0:
            args["cpu_cores"] = cpu_count()

    return args
