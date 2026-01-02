import os
import re
import sys
import time
import argparse

from typing import Sequence
from multiprocessing import cpu_count
from logic.src.utils.functions import parse_softmax_temperature
from logic.src.utils.definitions import (
    MAP_DEPOTS, WASTE_TYPES, STATS_FUNCTION_MAP,
    SUB_NET_ENCS, PRED_ENC_MODELS, ENC_DEC_MODELS, 
    OPERATION_MAP, APP_STYLES, FS_COMMANDS, TEST_MODULES,
)


class ConfigsParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser to handle string-based nargs correctly.
    """
    def _str_to_nargs(self, nargs):
        if isinstance(nargs, Sequence) and len(nargs) == 1:    
            return nargs[0].split() if isinstance(nargs[0], str) else nargs
        else:
            return nargs

    def _process_args(self, namespace):
        for action in self._actions:
            if action.nargs is not None:
                if action.dest == 'help':
                    continue

                # Check if the argument has nargs and process it
                value = getattr(namespace, action.dest)
                if value is not None:
                    transformed_value = self._str_to_nargs(value)
                    setattr(namespace, action.dest, transformed_value)

    def parse_command(self, args=None):
        if args is None:
            args = sys.argv[1:]

        namespace = super().parse_args(args)
        return getattr(namespace, 'command', None)
    
    def parse_process_args(self, args=None, command=None):
        if args is None:
            args = sys.argv[1:]

        # Get all actions to iterate over: main actions + current subparser actions
        actions_to_check = list(self._actions)
        
        # Try to find the actions of the specific subparser command
        command_name = None
        if args and not args[0].startswith('-'):
            command_name = args[0]
        
        if command_name:
            # Find the SubParsersAction to get the sub-parser object
            subparsers_action = next((a for a in actions_to_check if isinstance(a, argparse._SubParsersAction)), None)
            
            if subparsers_action and command_name in subparsers_action.choices:
                sub_parser = subparsers_action.choices[command_name]
                # Add subparser actions to the list to be checked
                actions_to_check.extend(sub_parser._actions)


        for action in actions_to_check:
            if action.dest == 'help':
                continue

            # Split strings with whitespace for nargs
            if action.nargs is not None and action.type is not None:
                opts = action.option_strings
                idx = next((i for i, x in enumerate(args) if x in opts), None)
                if idx is not None and (idx + 1) < len(args):
                    arg_val = args[idx+1]
                    # Check if the argument value is a single string and not an option flag
                    if isinstance(arg_val, str) and not arg_val.startswith('-'):
                        arg_parts = arg_val.split()
                        if len(arg_parts) > 1:
                            args[idx+1:idx+2] = arg_parts

        subnamespace = super().parse_args(args)
        parsed_args_dict = vars(subnamespace)
        filtered_args = {
            key: value if value != "" else None
            for key, value in parsed_args_dict.items() 
        }

        command = filtered_args.pop('command')
        return command, filtered_args if command is None else filtered_args
    
    def error_message(self, message, print_help=True):
        print(message, end=' ')
        if print_help:
            self.print_help()
        raise
    

class LowercaseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            values = str(values).lower()
        setattr(namespace, self.dest, values)


class StoreDictKeyPair(argparse.Action):
    """
    Custom action to parse arguments in the form key=value into a dictionary.
    Usage: --arg key1=value1 key2=value2
    """
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            if "=" in kv:
                k, v = kv.split("=", 1)
                my_dict[k] = v
            else:
                # Handle cases where formatting is incorrect
                raise argparse.ArgumentError(self, f"Could not parse argument '{kv}' as key=value format")
        setattr(namespace, self.dest, my_dict)


def UpdateFunctionMapActionFactory(inplace=False):
    """Factory function to create custom Action with flag"""
    class UpdateFunctionMapAction(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            super().__init__(option_strings, dest, nargs=nargs, **kwargs)
            self.inplace = inplace
        
        def __call__(self, parser, namespace, values, option_string=None):
            if values is not None:
                if self.inplace:
                    values = OPERATION_MAP.get(str(values).replace(' ', ''), None)
                else:
                    values = STATS_FUNCTION_MAP.get(str(values).replace(' ', ''), None)
            if values is None:
                raise ValueError(f"Invalid update function: {values}")
            setattr(namespace, self.dest, values)
    return UpdateFunctionMapAction

# ==============================================================================
# 
# ARGUMENT BUILDER FUNCTIONS
#
# ==============================================================================
def add_train_args(parser):
    """
    Adds all arguments related to training to the given parser.
    """
    # Data
    parser.add_argument('--problem', default='wcvrp', help="The problem to solve")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--edge_threshold', default='0', type=str, help="How many of all possible edges to consider")
    parser.add_argument('--edge_method', type=str, default=None, help="Method for getting edges ('dist'|'knn')")
    parser.add_argument('--batch_size', type=int, default=256, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=128_000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=0, 
                        help='Number of instances used for reporting validation performance (0 to deactivate validation)')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--eval_batch_size', type=int, default=256, help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--train_dataset', type=str, default=None, help='Name of dataset to use for training')

    # WSmart+ Route
    parser.add_argument('--eval_time_days', type=int, default=1, help='Number of days to perform validation (if train_time=True)')
    parser.add_argument('--train_time', action='store_true', help='Set to train the model over multiple days on the same graphs (n_days=n_epochs)')
    parser.add_argument('--area', type=str, default='riomaior', help='County area of the bins locations')
    parser.add_argument('--waste_type', type=str, default='plastic', help='Type of waste bins selected for the optimization problem')
    parser.add_argument('--focus_graph', default=None, help='Path to the file with the coordinates of the graph to focus on')
    parser.add_argument('--focus_size', type=int, default=0, help='Number of focus graphs to include in the training data')
    parser.add_argument('--eval_focus_size', type=int, default=0, help='Number of focus graphs to include in the validation data')
    parser.add_argument('--distance_method', type=str, default='ogd', help="Method to compute distance matrix")
    parser.add_argument('--dm_filepath', type=str, default=None, help="Path to the file to read/write the distance matrix from/to")
    parser.add_argument('--waste_filepath', type=str, default=None, help="Path to the file to read the waste fill for each day from")
    parser.add_argument('--vertex_method', type=str, default="mmn", help="Method to transform vertex coordinates "
                        "'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'")

    # Model
    parser.add_argument('--model', default='am', help="Model: 'am'|'tam'|'ddam'")
    parser.add_argument('--encoder', default='gat', help="Encoder: 'gat'|gac'|'tgc'|'ggac'|'gcn'|'mlp'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='Number of layers in the encoder/critic network')
    parser.add_argument('--n_encode_sublayers', type=int, default=None, help='Number of sublayers in the encoder network')
    parser.add_argument('--n_predict_layers', type=int, default=None, help='Number of layers in the predictor network')
    parser.add_argument('--n_decode_layers', type=int, default=None, help='Number of layers in the decoder network')
    parser.add_argument('--temporal_horizon', type=int, default=0, help="Number of previous days/epochs to take into account in predictions")
    parser.add_argument('--tanh_clipping', type=float, default=10., help='Clip the parameters to within +- this value using tanh. '
                            'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='instance', help="Normalization type: 'instance'|'layer'|'batch'|'group'|'local_response'|None")
    parser.add_argument('--learn_affine', action='store_false', help="Disable learnable affine transformation during normalization")
    parser.add_argument('--track_stats', action='store_true', help="Track statistics during normalization")
    parser.add_argument('--epsilon_alpha', type=float, default=1e-05, help="Epsilon (or alpha multiplicative, for LocalResponseNorm) value")
    parser.add_argument('--momentum_beta', type=float, default=0.1, help="Momentum (or beta exponential, for LocalResponseNorm) value")
    parser.add_argument('--lrnorm_k', type=float, default=None, help="Additive factor for LocalResponseNorm")
    parser.add_argument('--gnorm_groups', type=int, default=4, help="Number of groups to separate channels into for GroupNorm")
    parser.add_argument('--activation', default='gelu', help="Activation function: 'gelu'|'gelu_tanh'|'tanh'|'tanhshrink'|'mish'|'hardshrink'|'hardtanh'|'hardswish'|'glu'|"
                        "'relu'|'leakyrelu'|'silu'|'selu'|'elu'|'celu'|'prelu'|'rrelu'|'sigmoid'|'logsigmoid'|'hardsigmoid'|'threshold'|'softplus'|'softshrink'|'softsign'")
    parser.add_argument('--af_param', type=float, default=1.0, help="Parameter for the activation function formulation")
    parser.add_argument('--af_threshold', type=float, default=None, help="Threshold value for the activation function")
    parser.add_argument('--af_replacement', type=float, default=None, help="Replacement value for the activation function (above/below threshold)")
    parser.add_argument('--af_nparams', type=int, default=3, help="Number of parameters a for the Parametric ReLU (PReLU) activation")
    parser.add_argument('--af_urange', type=float, nargs='+', default=[0.125, 1/3], help="Range for the uniform distribution of the Randomized Leaky ReLU (RReLU) activation")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument('--aggregation_graph', default='avg', help="Graph embedding aggregation function: 'sum'|'avg'|'max'|None")
    parser.add_argument('--aggregation', default='sum', help="Node embedding aggregation function: 'sum'|'avg'|'max'")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--mask_inner', action='store_false', help="Mask inner values during decoding")
    parser.add_argument('--mask_logits', action='store_false', help="Mask logits during decoding")
    parser.add_argument('--mask_graph', action='store_true', help="Mask next node selection (using edges) during decoding")
    parser.add_argument('--spatial_bias', action='store_true', help="Enable spatial bias in decoder attention")
    parser.add_argument('--spatial_bias_scale', type=float, default=1.0, help="Scaling factor for the spatial bias penalty")
    parser.add_argument('--entropy_weight', type=float, default=0.0, help="Weight for the entropy regularization")
    parser.add_argument('--imitation_weight', type=float, default=0.0, help="Initial weight for the imitation loss guidance")
    parser.add_argument('--imitation_decay', type=float, default=1.0, help="Decay factor for the imitation weight")
    parser.add_argument('--imitation_decay_step', type=int, default=1, help="Number of epochs after which to apply imitation decay")
    parser.add_argument('--two_opt_max_iter', type=int, default=0, help='Maximum number of iterations for 2-opt refinement in Look-Ahead update')

    # Training
    parser.add_argument('--n_epochs', type=int, default=25, help='The number of epochs to train')
    parser.add_argument('--epoch_start', type=int, default=0, help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic_value', type=float, default=1e-4, help="Set the learning rate for the critic/value network")
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum L2 norm for gradient clipping (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--enable_scaler', action='store_true', help='Enables CUDA scaler for automatic mixed precision training')
    parser.add_argument('--exp_beta', type=float, default=0.8, help='Exponential moving average baseline decay')
    parser.add_argument('--baseline', default=None, help="Baseline to use: 'rollout'|'critic'|'exponential'|'pomo'|None")
    parser.add_argument('--bl_alpha', type=float, default=0.05, help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=-1, help='Number of epochs to warmup the baseline, ' 
                            'None means 1 for rollout (exponential used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--checkpoint_encoder', action='store_true', help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None, help='Shrink the batch size if at least this many instances in the batch'
                            ' are finished to save memory (default None means no shrinking)')
    parser.add_argument('--pomo_size', type=int, default=0, help='Number of starting nodes for POMO (Policy Optimization with Multiple Optima)')
    parser.add_argument('--data_distribution', type=str, default=None, help='Data distribution to use during training,'
                            ' defaults and options depend on problem. "empty"|"const"|"unif"|"dist"|"gamma[1-4]|"emp""')
    parser.add_argument('--accumulation_steps', type=int, default=1, 
                            help='Gradient accumulation steps during training (effective batch_size = batch_size * accumulation_steps)')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--post_processing_epochs', type=int, default=0, help='The number of epochs for post-processing')
    parser.add_argument('--lr_post_processing', type=float, default=0.001, help="Set the learning rate for post-processing")
    parser.add_argument('--efficiency_weight', type=float, default=0.8, help="Weight for the efficiency in post-processing")
    parser.add_argument('--overflow_weight', type=float, default=0.2, help="Weight for the bin overflows in post-processing")

    # Optimizer and learning rate scheduler
    parser.add_argument('--optimizer', type=str, default='rmsprop', 
                            help="Optimizer: 'adam'|'adamax'|'adamw'|'radam'|'nadam'|'sadam'|'adadelta'|'adagrad'|'rmsprop'|'rprop'|'lbfgs'|'asgd'|'sgd'")
    parser.add_argument('--lr_scheduler', type=str, default='lambda', 
                            help="Learning rate scheduler: 'exp'|'step'|'mult'|'lambda'|'const'|'poly'|'multistep'|'cosan'|'linear'|'cosanwr'|'plateau'")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--lr_min_value', type=float, default=0.0, help='Minimum learning rate for CosineAnnealingLR|CosineAnnealingWarmRestarts|ReduceLROnPlateau')
    parser.add_argument('--lr_min_decay', type=float, default=1e-8, help='Minimum decay applied to learning rate for LinearLR|ReduceLROnPlateau')
    parser.add_argument('--lrs_step_size', type=int, default=1, help='Period of learning rate decay for StepLR')
    parser.add_argument('--lrs_total_steps', type=int, default=5, help='Number of steps that the scheduler updates the lr for ConstantLR|LinearLR|PolynomialLR')
    parser.add_argument('--lrs_restart_steps', type=int, default=7, help='Number of steps until the first restart for CosineAnnealingWarmRestarts')
    parser.add_argument('--lrs_rfactor', type=int, default=2, help='A factor that, after a restart, increases the steps for the next restart for CosineAnnealingWarmRestarts.')
    parser.add_argument('--lrs_milestones', type=int, nargs='+', default=[7, 14, 21, 28], help='List of epoch indices (must be increasing) for MultiStepLR.')
    parser.add_argument('--lrs_mode', type=str, default='min', help='Scheduler mode for ReduceLROnPlateau')
    parser.add_argument('--lrs_dfactor', type=float, default=0.1, help='A factor by which the learning rate will be decreased for ReduceLROnPlateau')
    parser.add_argument('--lrs_patience', type=int, default=10, help='Number of epochs with no improvement after which the learning rate will be updated for ReduceLROnPlateau')
    parser.add_argument('--lrs_thresh', type=float, default=1e-4, help='Threshold for measuring the new optimum, to only focus on significant changes for ReduceLROnPlateau')
    parser.add_argument('--lrs_thresh_mode', type=str, default='rel', choices=['rel', 'abs'], help='Scheduler threshold mode for ReduceLROnPlateau')
    parser.add_argument('--lrs_cooldown', type=int, default=0, help='Number of epochs to wait before resuming normal operation after lr has been reduced for ReduceLROnPlateau')

    # Cost function weights
    parser.add_argument('--w_waste', type=float, default=None, help="Weight for the waste collected")
    parser.add_argument('--w_length', '--w_len', type=float, default=None, help="Weight for the route length")
    parser.add_argument('--w_overflows', '--w_over', type=float, default=None, help="Weight for the number of overflows")
    parser.add_argument('--w_lost', type=float, default=None, help="Weight for the amount of waste lost when bins overflow")
    parser.add_argument('--w_penalty', '--w_pen', type=float, default=None, help="Weight for the penalty")
    parser.add_argument('--w_prize', type=float, default=None, help="Weight for the prize")

    # Output
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default=None, help='Name to identify the run')
    parser.add_argument('--output_dir', default='assets/model_weights', help='Directory to write output models to')
    parser.add_argument('--checkpoints_dir', default='model_weights', help='Directory to write checkpoints to')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='Save checkpoint every n epochs, 0 to save no checkpoints')
    parser.add_argument('--wandb_mode', default='offline', help="Weights and biases mode: 'online'|'offline'|'disabled'|None")
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    return parser

def add_mrl_train_args(parser):
    """
    Adds arguments for Meta-Reinforcement Learning (inherits from train_args).
    """
    parser = add_train_args(parser)
    
    # MRL specific args
    parser.add_argument('--mrl_method', type=str, default='cb', choices=['tdl', 'rwa', 'cb', 'morl', 'hrl'], help='Method to use for Meta-Reinforcement Learning')
    parser.add_argument('--mrl_history', type=int, default=10, help="Number of previous days/epochs to take into account during Meta-Reinforcement Learning")
    parser.add_argument('--mrl_range', type=float, nargs='+', default=[0.01, 5.0], help="Maximum and minimum values for Meta-Reinforcement Learning with dynamic hyperparameters")
    parser.add_argument('--mrl_exploration_factor', type=float, default=2.0, help="Factor that controls the exploration vs. exploitation balance")
    parser.add_argument('--mrl_lr', type=float, default=1e-3, help="Set the learning rate for Meta-Reinforcement Learning")
    parser.add_argument('--mrl_embedding_dim', type=int, default=128, help='Dimension of input embedding for Reward Weight Adjustment model')
    parser.add_argument('--mrl_step', type=int, default=100, help='Update every mrl_step steps')
    parser.add_argument('--mrl_batch_size', type=int, default=256, help="Batch size to use for Meta-Reinforcement Learning")
    parser.add_argument('--hrl_threshold', type=float, default=0.9, help="Set the critical threshold for Hierarchical Reinforcement Learning PPO")
    parser.add_argument('--hrl_epochs', type=int, default=4, help="Number of epochs to use for Hierarchical Reinforcement Learning PPO")
    parser.add_argument('--hrl_clip_eps', type=float, default=0.2, help="Set the clip epsilon for Hierarchical Reinforcement Learning PPO")
    parser.add_argument('--shared_encoder', action='store_true', default=True, help="Set to share the encoder between worker and manager in HRL")
    parser.add_argument('--global_input_dim', type=int, default=2, help="Dimension of global input for HRL Manager")
    parser.add_argument('--gat_hidden', type=int, default=128, help="Hidden dimension for GAT Manager")
    parser.add_argument('--lstm_hidden', type=int, default=64, help="Hidden dimension for LSTM in GAT Manager")
    parser.add_argument('--gate_prob_threshold', type=float, default=0.5, help="Threshold for routing decision gate")
    parser.add_argument('--tdl_lr_decay', type=float, default=1.0, help='Learning rate decay for Temporal Difference Learning')
    parser.add_argument('--cb_exploration_method', type=str, default='ucb', help="Method for exploration in Contextual Bandits: 'ucb'|'thompson_sampling'|'epsilon_greedy'")
    parser.add_argument('--cb_num_configs', type=int, default=10, help="Number of weight configurations to generate in Contextual Bandits")
    parser.add_argument('--cb_context_features', type=str, nargs='+', default=['waste', 'overflow', 'length', 'visited_ratio', 'day'], help="Features for Contextual Bandits")
    parser.add_argument('--cb_features_aggregation', default='avg', help="Context features aggregation function in Contextual Bandits: 'sum'|'avg'|'max'")
    parser.add_argument('--cb_epsilon_decay', type=float, default=0.995, help="Decay factor for epsilon (=exploration_factor in 'epsilon_greedy')")
    parser.add_argument('--cb_min_epsilon', type=float, default=0.01, help="Minimum value for epsilon (=exploration_factor in 'epsilon_greedy')")
    parser.add_argument('--morl_objectives', type=str, nargs='+', default=['waste_efficiency', 'overflow_rate'], help="Objectives for Multi-Objective RL")
    parser.add_argument('--morl_adaptation_rate', type=float, default=0.1, help="Adaptation rate in Multi-Objective RL")
    parser.add_argument('--rwa_model', type=str, default='rnn', choices=['rnn'], help='Neural network to use for Reward Weight Adjustment')
    parser.add_argument('--rwa_optimizer', type=str, default='rmsprop', help="Optimizer: 'adamax'|'adam'|'adamw'|'radam'|'nadam'|'rmsprop'")
    parser.add_argument('--rwa_update_step', type=int, default=100, help='Update Reward Weight Adjustment weights every rwa_update_step steps')
    
    # HRL PID and Reward Shaping
    parser.add_argument('--hrl_pid_target', type=float, default=0.05, help="Target overflow rate for PID control")
    parser.add_argument('--hrl_kp', type=float, default=50.0, help="Kp factor for HRL PID overflow control")
    parser.add_argument('--hrl_ki', type=float, default=5.0, help="Ki factor for HRL PID overflow control")
    parser.add_argument('--hrl_kd', type=float, default=0.0, help="Kd factor for HRL PID overflow control")
    parser.add_argument('--hrl_lambda_overflow_initial', type=float, default=1000.0, help="Initial lambda weight for overflows")
    parser.add_argument('--hrl_lambda_overflow_min', type=float, default=100.0, help="Minimum lambda weight for overflows")
    parser.add_argument('--hrl_lambda_overflow_max', type=float, default=2000.0, help="Maximum lambda weight for overflows")
    parser.add_argument('--hrl_lambda_waste', type=float, default=300.0, help="Reward weight for collected waste")
    parser.add_argument('--hrl_lambda_cost', type=float, default=0.1, help="Penalty weight for route cost/distance")
    parser.add_argument('--hrl_lambda_pruning', type=float, default=5.0, help="Penalty weight for masking too many nodes")
    parser.add_argument('--hrl_reward_scale', type=float, default=0.0001, help="Final scaling factor for HRL rewards")
    
    # HRL Training Hyperparameters
    parser.add_argument('--hrl_gamma', type=float, default=0.95, help="Discount factor for manager PPO update")
    parser.add_argument('--hrl_lambda_mask_aux', type=float, default=50.0, help="Weight for mask auxiliary loss")
    parser.add_argument('--hrl_entropy_coef', type=float, default=0.2, help="Entropy coefficient for manager PPO update")
    
    return parser

def add_hp_optim_args(parser):
    """
    Adds arguments for Hyper-Parameter Optimization (inherits from train_args).
    """
    parser = add_train_args(parser)
    
    # HPO specific args
    parser.add_argument('--hop_method', type=str, default='dehbo', choices=['dea', 'bo', 'hbo', 'rs', 'gs', 'dehbo'],
                                help='Method to use for hyperparameter optimization')
    parser.add_argument('--hop_range', type=float, nargs='+', default=[0.0, 2.0], help="Maximum and minimum values for hyperparameter optimization")
    parser.add_argument('--hop_epochs', type=int, default=7, help='The number of epochs to optimize hyperparameters')
    parser.add_argument('--metric', type=str, default="val_loss", choices=[
        'loss', 'val_loss', 'mean_reward', 'mae', 'mse', 'rmse', 'episode_reward_mean', 'kg/km', 'overflows', 'both'], help='Metric to optimize')
    
    # ===== Bayesian Optimization (BO) =====
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials for Optuna optimization')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout for Optuna optimization (in seconds)')
    parser.add_argument('--n_startup_trials', type=int, default=5, help='Number of trials to run before pruning starts')
    parser.add_argument('--n_warmup_steps', type=int, default=3, help='Number of epochs to wait before pruning can happen in each trial')
    parser.add_argument('--interval_steps', type=int, default=1, help='Pruning is evaluated every this many epochs')

    # ===== Distributed Evolutionary Algorithm (DEA) =====
    parser.add_argument('--eta', type=float, default=10.0, help="Controls the spread of the genetic mutations (higher = slower changes)")
    parser.add_argument('--indpb', type=float, default=0.2, help="Probability of mutating each gene of an individual in the population")
    parser.add_argument('--tournsize', type=int, default=3, help="Number of individuals to fight to be selected for reproduction")
    parser.add_argument('--cxpb', type=float, default=0.7, help="Probability of crossover between two parents (higher = faster convergence + less diversity)")
    parser.add_argument('--mutpb', type=float, default=0.2, help="Probability of an individual being mutated after crossover")
    parser.add_argument('--n_pop', type=int, default=20, help="Starting population for evolutionary algorithms")
    parser.add_argument('--n_gen', type=int, default=10, help="Number of generations to evolve")

    # ===== Differential Evolutionary Hyperband Optimization (DEHBO) =====
    parser.add_argument('--fevals', type=int, default=100, help='Number of function evaluations')

    # Ray Tune framework hyperparameters
    parser.add_argument('--cpu_cores', type=int, default=1, help="Number of CPU cores to use for hyperparameter optimization (0 uses all available cores)")
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity level for Hyperband and Random Search (0-3)')
    parser.add_argument('--train_best', action='store_true', default=True, help='Train final model with best hyperparameters')
    parser.add_argument('--local_mode', action='store_true', help='Run ray in local mode (for debugging)')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of times to sample from the hyperparameter space')

    # ===== Hyperband Optimization (HBO) - Ray Tune =====
    parser.add_argument('--max_tres', type=int, default=14, help='Maximum resources (timesteps) per trial')
    parser.add_argument('--reduction_factor', type=int, default=3, help='Reduction factor for Hyperband')

    # ===== Random Search (RS) - Ray Tune =====
    parser.add_argument('--max_failures', type=int, default=3, help='Maximum trial failures before stopping')

    # ===== Grid Search (GS) - Ray Tune =====
    parser.add_argument('--grid', type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5, 2.0], help='Hyperparameter values to try in grid search')
    parser.add_argument('--max_conc', type=int, default=4, help='Maximum number of concurrent trials for Ray Tune')
    return parser

def add_gen_data_args(parser):
    """
    Adds all arguments related to data generation to the given parser.
    """
    parser.add_argument("--name", type=str, help="Name to identify dataset")
    parser.add_argument("--filename", default=None, help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='datasets', help="Create datasets in data")
    parser.add_argument("--problem", type=str, default='all', help="Problem: 'vrpp'|'wcvrp'|'swcvrp'|'all'")
    parser.add_argument("--mu", type=float, default=None, nargs='+', help="Mean of Gaussian noise (implies Gaussian noise generation if set)")
    parser.add_argument('--sigma', type=float, nargs='+', default=0.6, help="Variance of Gaussian noise")
    parser.add_argument('--data_distributions', nargs='+', default=['all'], help="Distributions to generate for problems")
    parser.add_argument("--dataset_size", type=int, default=128_000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100], help="Sizes of problem instances")
    parser.add_argument('--penalty_factor', type=float, default=3.0, help="Penalty factor for problems")
    parser.add_argument("-f", action='store_true', dest="overwrite", help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--n_epochs', type=int, default=1, help='The number of epochs to generate data for')
    parser.add_argument('--epoch_start', type=int, default=0, help='Start at epoch #')
    parser.add_argument('--dataset_type', type=str, choices=['train', 'train_time', 'test_simulator'], help='Set type of dataset to generate')
    parser.add_argument('--area', type=str, default='riomaior', help='County area of the bins locations')
    parser.add_argument('--waste_type', type=str, default='plastic', help='Type of waste bins selected for the optimization problem')
    parser.add_argument('--focus_graphs', nargs='+', default=None, help='Path to the files with the coordinates of the graphs to focus on')
    parser.add_argument('--focus_size', type=int, default=0, help='Number of focus graphs to include in the data')
    parser.add_argument('--vertex_method', type=str, default="mmn", help="Method to transform vertex coordinates "
                                "'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'")
    return parser

def add_eval_args(parser):
    """
    Adds all arguments related to evaluation to the given parser.
    """
    parser.add_argument("--datasets", type=str, nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', dest="overwrite", help="Set true to overwrite")
    parser.add_argument("-o", "--output_filename", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=12_800, help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0, help='Offset where to start in dataset')
    parser.add_argument('--eval_batch_size', type=int, default=256, help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--decode_type', type=str, default='greedy', help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, nargs='+', help='Sizes of beam to use for beam search (or number of samples for sampling), '
                            '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str, help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1, help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use')
    parser.add_argument('--data_distribution', type=str, default=None, help='Data distribution of the dataset')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=12_800, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of evaluation results directory")
    parser.add_argument('--multiprocessing', action='store_true', help='Use multiprocessing to parallelize over multiple GPUs')
    parser.add_argument('--graph_size', type=int, default=50, help="The size of the problem graph")
    parser.add_argument('--area', type=str, default='riomaior', help='County area of the bins locations')
    parser.add_argument('--waste_type', type=str, default='plastic', help='Type of waste bins selected for the optimization problem')
    parser.add_argument('--focus_graph', default=None, help='Path to the file with the coordinates of the graph to focus on')
    parser.add_argument('--focus_size', type=int, default=0, help='Number of focus graphs to include in the training data')
    parser.add_argument('--edge_threshold', default='0', type=str, help="How many of all possible edges to consider")
    parser.add_argument('--edge_method', type=str, default=None, help="Method for getting edges ('dist'|'knn')")
    parser.add_argument('--distance_method', type=str, default='ogd', help="Method to compute distance matrix")
    parser.add_argument('--dm_filepath', type=str, default=None, help='Path to the distance matrix file')
    parser.add_argument('--vertex_method', type=str, default="mmn", help="Method to transform vertex coordinates "
                            "'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'")
    parser.add_argument('--w_length', type=float, default=1.0, help='Weight for length in cost function')
    parser.add_argument('--w_waste', type=float, default=1.0, help='Weight for waste in cost function')
    parser.add_argument('--w_overflows', type=float, default=1.0, help='Weight for overflows in cost function')
    parser.add_argument('--problem', type=str, default='cwcvrp', help="Problem to evaluate ('wcvrp'|'cwcvrp'|'sdwcvrp'|'scwcvrp')")
    parser.add_argument('--encoder', type=str, default='gat', help="Encoder to use ('gat'|'gac'|'tgc'|'ggac')")
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    return parser

def add_test_sim_args(parser):
    """
    Adds all arguments related to the test simulator to the given parser.
    """
    parser.add_argument('--policies', type=str, nargs='+', required=True, help="Name of the policy(ies) to test on the WSR simulator")
    parser.add_argument('--gate_prob_threshold', type=float, default=0.5, help="Probability threshold for gating decisions (default: 0.5)")
    parser.add_argument('--mask_prob_threshold', type=float, default=0.5, help="Probability threshold for mask decisions (default: 0.5)")
    parser.add_argument('--data_distribution', '--dd', type=str, default='gamma1', help="Distribution to generate the bins daily waste fill")
    parser.add_argument('--problem', default='vrpp', help="The problem the model was trained to solve")
    parser.add_argument('--size', type=int, default=50, help="The size of the problem graph")
    parser.add_argument('--days', type=int, default=31, help="Number of days to run the simulation for")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output_dir', default='output', help="Name of WSR simulator test output directory")
    parser.add_argument('--checkpoint_dir', default='temp', help="Name of WSR simulator test runs checkpoint directory")
    parser.add_argument('--checkpoint_days', '--cpd', type=int, default=5, help='Save checkpoint every n days, 0 to save no checkpoints')
    parser.add_argument('--n_samples', type=int, default=1, help="Number of simulation samplings for each policy")
    parser.add_argument('--resume', action='store_true', help='Resume testing (relevant for saving results)')
    parser.add_argument('--pregular_level', '--lvl', type=int, nargs='+', help="Regular policy level")
    parser.add_argument('--plastminute_cf', '--cf', type=int, nargs='+', help="CF value for last minute/last minute and path policies")
    parser.add_argument('--lookahead_configs', '--lac', type=str, nargs='+', help="Parameter configuration for policy Look-Ahead and variants")
    parser.add_argument('--gurobi_param', '--gp', type=float, default=0.84, nargs='+', help='Param value for Gurobi VRPP policy '
                            '(higher = more conservative with regards to amount of overflows)')
    parser.add_argument('--hexaly_param', '--hp', type=float, default=2.0, nargs='+', help='Param value for Hexaly optimizer policy '
                            '(higher = more conservative with regards to amount of overflows)')
    parser.add_argument('--cpu_cores', '--cc', type=int, default=0, help="Number of max CPU cores to use (0 uses all available cores)")
    parser.add_argument('--n_vehicles', type=int, default=1, help="Number of vehicles")
    parser.add_argument('--area', type=str, default='riomaior', help='County area of the bins locations')
    parser.add_argument('--waste_type', type=str, default='plastic', help='Type of waste bins selected for the optimization problem')
    parser.add_argument('--bin_idx_file', type=str, default=None, help="File with the indices of the bins to use in the simulation")
    parser.add_argument('--decode_type', '--dt', type=str, default='greedy', help='Decode type, greedy or sampling')
    parser.add_argument('--temperature', type=parse_softmax_temperature, default=1, help="Softmax temperature (sampling or bs)")
    parser.add_argument('--edge_threshold', '--et', default='0', type=str, help='How many of all possible edges to consider')
    parser.add_argument('--edge_method', '--em', type=str, default=None, help="Method for getting edges ('dist'|'knn')")
    parser.add_argument('--vertex_method', '--vm', type=str, default="mmn", help="Method to transform vertex coordinates "
                            "'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'")
    parser.add_argument('--distance_method', '--dm', type=str, default='ogd', help="Method to compute distance matrix")
    parser.add_argument('--dm_filepath', '--dmf', type=str, default=None, help="Path to the file to read/write the distance matrix from/to")
    parser.add_argument('--waste_filepath', type=str, default=None, help="Path to the file to read the waste fill for each day from")
    parser.add_argument('--noise_mean', type=float, default=0.0, help="Mean of Gaussian noise to inject into observed bin levels")
    parser.add_argument('--noise_variance', type=float, default=0.0, help="Variance of Gaussian noise to inject into observed bin levels")
    parser.add_argument('--run_tsp', action='store_true', help="Activate fast_tsp for all policies.")
    parser.add_argument('--spatial_bias', action='store_true', help="Enable spatial bias in decoder attention")
    parser.add_argument('--two_opt_max_iter', type=int, default=0, help='Maximum number of 2-opt iterations')
    parser.add_argument('--cache_regular', action='store_false', help="Deactivate caching for policy regular.")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--server_run', action='store_true', help="Simulation will be executed in a remote server")
    parser.add_argument('--env_file', type=str, default='vars.env', help="Name of the file that contains the environment variables")
    parser.add_argument('--gplic_file', type=str, default=None, help="Name of the file that contains the license to use for Gurobi")
    parser.add_argument('--hexlic_file', type=str, default=None, help="Name of the file that contains the license to use for Gurobi")
    parser.add_argument('--symkey_name', type=str, default=None, help="Name of the cryptographic key used to access the API keys")
    parser.add_argument('--gapik_file', type=str, default=None, help="Name of the file that contains the key to use for the Google API")
    parser.add_argument('--real_time_log', action='store_true', help="Activate real time results window")
    parser.add_argument('--stats_filepath', type=str, default=None, help="Path to the file to read the statistics from")
    parser.add_argument('--model_path', action=StoreDictKeyPair, default=None, nargs='+', 
                        help="Path to the directory where the model(s) is/are stored (format: name=path)")
    return parser

def add_files_update_args(parser):
    """Adds file system 'update' sub-command arguments."""
    parser.add_argument('--target_entry', type=str, help="Path to the file to the file system entry we want to update")
    parser.add_argument('--output_key', type=str, default=None, help="Key of the values we want to update in the files")
    parser.add_argument('--filename_pattern', type=str, default=None, help="Pattern to match names of files to update (target_entry must be directory)")
    parser.add_argument('--update_operation', type=str, default=None, action=UpdateFunctionMapActionFactory(inplace=True), help="Operation to update the file values")
    parser.add_argument('--update_value', type=float, default=0.0, help="Value for the update operation")
    parser.add_argument('--update_preview', action='store_true', help="Preview how files/directories will look like after the update")
    parser.add_argument('--stats_function', type=str, default=None, action=UpdateFunctionMapActionFactory(), help="Function to perform over the file values")
    parser.add_argument('--output_filename', type=str, default=None, help="Name of the file we want to save the output values to")
    parser.add_argument('--input_keys', type=str, default=(None, None), nargs='*', help="Key(s) of the values we want to use as input to update the other key in the files")
    return parser

def add_files_delete_args(parser):
    """Adds file system 'delete' sub-command arguments."""
    parser.add_argument('--wandb', action="store_false", help="Flag to delete the train wandb log directory")
    parser.add_argument('--log_dir', default='logs', help='Directory of train logs')
    parser.add_argument('--log', action="store_false", help="Flag to delete the train log directory")
    parser.add_argument('--output_dir', default='model_weights', help='Directory to write output models to')
    parser.add_argument('--output', action="store_false", help="Flag to delete the train output models directory")
    parser.add_argument("--data_dir", default='datasets', help="Directory of generated datasets")
    parser.add_argument('--data', action="store_true", help="Flag to delete the datasets directory")
    parser.add_argument('--eval_dir', default='results', help="Name of the evaluation results directory")
    parser.add_argument('--eval', action="store_true", help="Flag to delete the evaluation results directory")
    parser.add_argument('--test_dir', default='output', help="Name of the WSR simulator test output directory")
    parser.add_argument('--test', action="store_true", help="Flag to delete the WSR simulator test output directory")
    parser.add_argument('--test_checkpoint_dir', default='temp', help="Name of WSR simulator test runs checkpoint directory")
    parser.add_argument('--test_checkpoint', action="store_true", help="Flag to delete the WSR simulator test runs checkpoint directory")
    parser.add_argument('--cache', action="store_true", help="Flag to delete the cache directories")
    parser.add_argument('--delete_preview', action='store_true', help="Preview which files/directories will be removed")
    return parser

def add_files_crypto_args(parser):
    """Adds file system 'cryptography' sub-command arguments."""
    parser.add_argument('--symkey_name', type=str, default=None, help="Name of the key for the files to save the salt and key/hash parameters to")
    parser.add_argument('--env_file', type=str, default='vars.env', help="Name of the file that contains the environment variables")
    parser.add_argument('--salt_size', type=int, default=16)
    parser.add_argument('--key_length', type=int, default=32)
    parser.add_argument('--hash_iterations', type=int, default=100_000)
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    return parser

def add_files_args(parser):
    """
    Adds all arguments related to file system operations (as subparsers).
    """
    files_subparsers = parser.add_subparsers(help="file system command", dest="fs_command", required=True)

    # Update file system entries
    update_parser = files_subparsers.add_parser('update', help='Update file system entries')
    add_files_update_args(update_parser)

    # Delete file system entries
    delete_parser = files_subparsers.add_parser('delete', help='Delete file system entries')
    add_files_delete_args(delete_parser)

    # Cryptography
    crypto_parser = files_subparsers.add_parser('cryptography', help='Perform cryptographic operations on file system entries')
    add_files_crypto_args(crypto_parser)
    return parser
    
def add_gui_args(parser):
    """
    Adds all arguments related to the GUI to the given parser.
    """
    parser.add_argument('--app_style', action=LowercaseAction, type=str, default="fusion", help="Style for the GUI application")
    parser.add_argument('--test_only', action='store_true', help="Test mode for the GUI (commands are only printed, not executed).")
    return parser
    
def add_test_suite_args(parser):
    """
    Adds all arguments related to the test suite to the given parser.
    """
    # Test selection
    parser.add_argument('-m', '--module', nargs='+', choices=list(TEST_MODULES.keys()), help='Specific test module(s) to run')
    parser.add_argument('-c', '--class', dest='test_class', help='Specific test class to run (e.g., TestTrainCommand)')
    parser.add_argument('-t', '--test', dest='test_method', help='Specific test method to run (e.g., test_train_default_parameters)')
    parser.add_argument('-k', '--keyword', help='Run tests matching the given keyword expression')
    parser.add_argument('--markers', help='Run tests matching the given marker expression')
    
    # Test execution options
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    parser.add_argument('--ff', '--failed-first', dest='failed_first', action='store_true', help='Run failed tests first')
    parser.add_argument('-x', '--exitfirst', dest='maxfail', action='store_const', const=1, help='Exit on first failure')
    parser.add_argument('--maxfail', type=int, help='Exit after N failures')
    parser.add_argument('--tb', choices=['auto', 'long', 'short', 'line', 'native', 'no'], default='auto', help='Traceback print mode')
    parser.add_argument('--capture', choices=['auto', 'no', 'sys', 'fd'], default='auto', help='Capture mode for output')
    parser.add_argument('-n', '--parallel', action='store_true', help='Run tests in parallel (requires pytest-xdist)')
    
    # Information commands
    parser.add_argument('-l', '--list', action='store_true', help='List all available test modules')
    parser.add_argument('--list-tests', action='store_true', help='List all tests in specified module(s) or all tests')
    parser.add_argument('--test-dir', default='tests', help='Directory containing test files (default: tests)')
    return parser


def get_main_parser():
    """
    Builds the main parser with all sub-commands (train, eval, gen_data, etc.).
    """
    parser = ConfigsParser(
        description="ML models and OR solvers for CO problems",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- Main Commands Subparsers ---
    subparsers = parser.add_subparsers(help='Main command', dest='command', required=True)

    # 1. Train Model (Base Training)
    train_parser = subparsers.add_parser('train', help='Train reinforcement learning model')
    add_train_args(train_parser)

    # 2. Meta RL Training
    mrl_parser = subparsers.add_parser('mrl_train', help='Train meta-reinforcement learning model')
    add_mrl_train_args(mrl_parser)

    # 3. Hyperparameter Optimization
    hp_optim_parser = subparsers.add_parser('hp_optim', help='Hyperparameter optimization')
    add_hp_optim_args(hp_optim_parser)

    # 4. Generate Data
    gen_data_parser = subparsers.add_parser('gen_data', help='Generate problem datasets')
    add_gen_data_args(gen_data_parser)

    # 5. Evaluate
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained models')
    add_eval_args(eval_parser)

    # 6. Test Simulator
    test_sim_parser = subparsers.add_parser('test_sim', help='Run WSR simulator test policies')
    add_test_sim_args(test_sim_parser)

    # 7. File System Tools (Requires inner sub-parsers)
    files_parser = subparsers.add_parser('file_system', help='File system manipulation tools')
    add_files_args(files_parser)

    # 8. Test Suite
    test_suite_parser = subparsers.add_parser('test_suite', help='Run program test suite')
    add_test_suite_args(test_suite_parser)

    # 9. GUI
    gui_parser = subparsers.add_parser('gui', help='Run the GUI application')
    add_gui_args(gui_parser)
    return parser


def parse_params():
    """
    Parses arguments, determines the command, and performs necessary validation.
    Returns: (command, validated_opts) where 'command' might be a tuple (comm, inner_comm)
    """
    parser = get_main_parser()
    
    try:
        # Parse arguments into a dictionary using the custom handler
        command, opts = parser.parse_process_args()

        # --- COMMAND-SPECIFIC VALIDATION AND POST-PROCESSING ---
        if command in ['train', 'mrl_train', 'hp_optim']:
            opts = validate_train_args(opts)
        elif command == 'gen_data':
            opts = validate_gen_data_args(opts)
        elif command == 'eval':
            opts = validate_eval_args(opts)
        elif command == 'test_sim':
            opts = validate_test_sim_args(opts)
        elif command == 'file_system':
            # This returns a tuple: (fs_command, validated_opts)
            command, opts = validate_file_system_args(opts)
            command = ('file_system', command) # Re-wrap for main() function handling
        elif command == 'gui':
            opts = validate_gui_args(opts)
        elif command == 'test_suite':
            opts = validate_test_suite_args(opts)
        return command, opts
    except (argparse.ArgumentError, AssertionError) as e:
        parser.error_message(f"Error: {e}", print_help=True)
    except Exception as e:
        parser.error_message(f"An unexpected error occurred: {e}", print_help=False)

# ==============================================================================
# 
# VALIDATION & POST-PROCESSING FUNCTIONS
#
# ==============================================================================

def validate_train_args(args):
    """
    Validates and post-processes arguments for train, mrl_train, and hp_optim.
    """
    args = args.copy()
    assert args['epoch_size'] % args['batch_size'] == 0, "Epoch size must be integer multiple of batch size!"
    
    if args.get('bl_warmup_epochs', -1) < 0:
        args['bl_warmup_epochs'] = 1 if args.get('baseline') == 'rollout' else 0

    assert (args['bl_warmup_epochs'] == 0) or (args.get('baseline') == 'rollout')
    
    if args.get('baseline') == 'pomo':
        assert args.get('pomo_size', 0) > 0, "pomo_size must be > 0 when using pomo baseline"
    
    if args.get('encoder') in SUB_NET_ENCS and args.get('n_encode_sublayers') is None:
        args['n_encode_sublayers'] = args['n_encode_layers']
    
    assert args.get('encoder') not in SUB_NET_ENCS or args.get('n_encode_sublayers', 0) > 0, \
    f"Must select a positive integer for 'n_encode_sublayers' arg for {args.get('encoder')} encoder"
    
    if args.get('model') in PRED_ENC_MODELS and args.get('n_predict_layers') is None:
        args['n_predict_layers'] = args['n_encode_layers']

    assert args.get('model') not in PRED_ENC_MODELS or args.get('n_predict_layers', 0) > 0, \
    f"Must select a positive integer for 'n_predict_layers' arg for {args.get('model')} model" 
    
    if args.get('model') in ENC_DEC_MODELS and args.get('n_decode_layers') is None:
        args['n_decode_layers'] = args['n_encode_layers']

    assert args.get('model') not in ENC_DEC_MODELS or args.get('n_decode_layers', 0) > 0, \
    f"Must select a positive integer for 'n_decode_layers' arg for {args.get('model')} model" 
    
    if args.get('run_name') is not None:
        args['run_name'] = "{}_{}".format(args['run_name'], time.strftime("%Y%m%dT%H%M%S"))
    else:
        args['run_name'] = "{}{}{}{}_{}".format(
            args.get('model', 'model'), args.get('encoder', 'enc'), 
            args.get('temporal_horizon', 0) if args.get('temporal_horizon', 0) > 0 else "",
            "_{}".format(args.get('data_distribution')) if args.get('data_distribution') is not None else "", 
            time.strftime("%Y%m%dT%H%M%S"))
    
    args['save_dir'] = os.path.join(
        args.get('checkpoints_dir', 'model_weights'),
        "{}_{}".format(args.get('problem', 'problem'), args.get('graph_size', 'size')),
        args['run_name']
    )

    args['final_dir'] = os.path.join(
        args.get('output_dir', 'assets/model_weights'),
        "{}{}{}{}".format(
            args.get('problem', 'problem'), args.get('graph_size', 'size'),
            "_{}".format(args.get('area')) if 'area' in args and args['area'] is not None else "", 
            "_{}".format(args.get('waste_type')) if 'waste_type' in args and args['waste_type'] is not None else ""
        ),
        args.get('data_distribution') or '',
        "{}{}{}".format(
            args.get('model', 'model'), args.get('encoder', 'enc'), 
            "_{}".format(args.get('mrl_method')) if 'mrl_method' in args and args.get('mrl_method') is not None else ""
        )
    )
    
    if 'area' in args and args['area'] is not None:
        args['area'] = re.sub(r'[^a-zA-Z]', '', args['area'].lower())
        assert args['area'] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(args['area'], MAP_DEPOTS.keys())

    if 'waste_type' in args and args['waste_type'] is not None:
        assert 'area' in args and args['area'] is not None
        args['waste_type'] = re.sub(r'[^a-zA-Z]', '', args['waste_type'].lower())
        assert args['waste_type'] in WASTE_TYPES.keys() or args['waste_type'] is None, \
        "Unknown waste type {}, available waste types: {}".format(args['waste_type'], WASTE_TYPES.keys())

    args['edge_threshold'] = float(args['edge_threshold']) if '.' in args['edge_threshold'] else int(args['edge_threshold'])
    
    if 'hop_method' in args: # hp_optim specific
        assert args.get('cpu_cores', 1) >= 0, "Number of CPU cores must be non-negative integer"
        assert args.get('cpu_cores', 1) <= cpu_count(), "Number of CPU cores to use cannot exceed system specifications"
        if args.get('cpu_cores') == 0:
            args['cpu_cores'] = cpu_count()
            
    return args

def validate_gen_data_args(args):
    """
    Validates and post-processes arguments for gen_data.
    """
    args = args.copy()
    assert 'filename' not in args or args['filename'] is None or (
        len(args.get('problem', [])) == 1 and len(args.get('graph_sizes', [])) == 1
        ), "Can only specify filename when generating a single dataset"
    
    if args['problem'] in ['all', 'swcvrp']:
        assert 'mu' in args and args['mu'] is not None, "Must specify mu when generating swcvrp datasets"
        assert 'sigma' in args and args['sigma'] is not None, "Must specify sigma when generating swcvrp datasets"
        assert len(args['mu']) == len(args['sigma']), "Must specify same number of mu and sigma values"

    assert 'focus_graphs' not in args or args['focus_graphs'] is None or len(args['focus_graphs']) == len(args.get('graph_sizes', []))
    
    if 'focus_graphs' not in args or args['focus_graphs'] is None: 
        args['focus_graphs'] = [None] * len(args.get('graph_sizes', []))
    else:
        args['area'] = re.sub(r'[^a-zA-Z]', '', args.get('area', '').lower())
        assert args['area'] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(args['area'], MAP_DEPOTS.keys())

    args['waste_type'] = re.sub(r'[^a-zA-Z]', '', args.get('waste_type', '').lower())
    assert args['waste_type'] in WASTE_TYPES.keys() or args['waste_type'] is None, \
    "Unknown waste type {}, available waste types: {}".format(args['waste_type'], WASTE_TYPES.keys())
    return args

def validate_eval_args(args):
    """
    Validates and post-processes arguments for eval.
    """
    args = args.copy()
    # Handle the -o alias for output_filename
    if 'output_filename' in args and args['output_filename'] is not None:
        args['o'] = args['output_filename']
    
    assert 'o' not in args or args['o'] is None or (
        len(args.get('datasets') or []) == 1 and len(args.get('width') or []) <= 1
        ), "Cannot specify result filename with more than one dataset or more than one width"
    
    args['area'] = re.sub(r'[^a-zA-Z]', '', args.get('area', '').lower())
    assert args['area'] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(args['area'], MAP_DEPOTS.keys())

    args['waste_type'] = re.sub(r'[^a-zA-Z]', '', args.get('waste_type', '').lower())
    assert args['waste_type'] in WASTE_TYPES.keys() or args['waste_type'] is None, \
    "Unknown waste type {}, available waste types: {}".format(args['waste_type'], WASTE_TYPES.keys())
    return args

def validate_test_sim_args(args):
    """
    Validates and post-processes arguments for test_sim.
    """
    args = args.copy()
    assert args.get('days', 0) >= 1, "Must run the simulation for 1 or more days"
    assert args.get('n_samples', 0) > 0, "Number of samples must be non-negative integer"

    args['area'] = re.sub(r'[^a-zA-Z]', '', args.get('area', '').lower())
    assert args['area'] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(args['area'], MAP_DEPOTS.keys())

    args['waste_type'] = re.sub(r'[^a-zA-Z]', '', args.get('waste_type', '').lower())
    assert args['waste_type'] in WASTE_TYPES.keys() or args['waste_type'] is None, \
    "Unknown waste type {}, available waste types: {}".format(args['waste_type'], WASTE_TYPES.keys())

    args['edge_threshold'] = float(args['edge_threshold']) if '.' in str(args.get('edge_threshold', '0')) else int(args.get('edge_threshold', '0'))
    
    assert args.get('cpu_cores', 0) >= 0, "Number of CPU cores must be non-negative integer"
    assert args.get('cpu_cores', 0) <= cpu_count(), "Number of CPU cores to use cannot exceed system specifications"
    if args.get('cpu_cores') == 0:
        args['cpu_cores'] = cpu_count()

    if args.get('plastminute_cf'):
        vals = args['plastminute_cf'] if isinstance(args['plastminute_cf'], list) else [args['plastminute_cf']]
        for cf in vals:
            assert cf > 0 and cf < 100, "Policy last minute CF must be between 0 and 100"
    if args.get('pregular_level'):
        vals = args['pregular_level'] if isinstance(args['pregular_level'], list) else [args['pregular_level']]
        for lvl in vals:
            assert lvl >= 1 and lvl <= args['days'], "Policy regular level must be between 1 and number of days, inclusive"
    if args.get('gurobi_param'):
        vals = args['gurobi_param'] if isinstance(args['gurobi_param'], list) else [args['gurobi_param']]
        for gp in vals:
            assert gp > 0, "Policy gurobi parameter must be greater than 0"
    if args.get('hexaly_param'):
        vals = args['hexaly_param'] if isinstance(args['hexaly_param'], list) else [args['hexaly_param']]
        for hp in vals:
            assert hp > 0, "Policy hexaly parameter must be greater than 0"
    if args.get('lookahead_configs'):
        vals = args['lookahead_configs'] if isinstance(args['lookahead_configs'], list) else [args['lookahead_configs']]
        for lac in vals:
            assert lac in ['a', 'b'], "Policy lookahead configuration must be 'a' or 'b'"
    return args

def validate_file_system_args(args):
    """
    Validates and post-processes arguments for file_system.
    Returns a tuple: (fs_command, validated_args)
    """
    args = args.copy()
    fs_comm = args.pop('fs_command', None)
    if fs_comm not in FS_COMMANDS:
        raise argparse.ArgumentError(None, "ERROR: unknown File System (inner) command " + str(fs_comm))

    assert not ('stats_function' in args and args['stats_function'] is not None) or \
        not ('update_operation' in args and args['update_operation'] is not None), \
        "'update_operation' and 'stats_function' arguments are mutually exclusive"
    
    return fs_comm, args

def validate_gui_args(args):
    """
    Validates and post-processes arguments for gui.src.
    """
    args = args.copy()
    assert args.get('app_style') in [None] + APP_STYLES, \
    f"Invalid application style '{args.get('app_style')}' - app_style value must be: {[None] + APP_STYLES}"
    return args

def validate_test_suite_args(args):
    """
    Validates and post-processes arguments for test_suite.
    """
    args = args.copy()
    return args
