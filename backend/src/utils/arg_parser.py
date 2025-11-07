import os
import re
import sys
import time
import argparse

from typing import Iterable
from multiprocessing import cpu_count
from backend.src.utils.functions import parse_softmax_temperature
from backend.src.utils.definitions import (
    MAP_DEPOTS, WASTE_TYPES, STATS_FUNCTION_MAP,
    SUB_NET_ENCS, PRED_ENC_MODELS, ENC_DEC_MODELS, 
    OPERATION_MAP, APP_STYLES, FS_COMMANDS, TEST_MODULES,
)


class ConfigsParser(argparse.ArgumentParser):
    def _str_to_nargs(self, nargs):
        if isinstance(nargs, Iterable) and len(nargs) == 1:    
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
    
    def parse_process_args(self, args=None):
        if args is None:
            args = sys.argv[1:]
        for action in self._actions:
            if action.dest == 'help':
                continue

            # Split strings with whitespace for nargs
            if action.nargs is not None and action.type is not None:
                opts = action.option_strings
                idx = next((i for i, x in enumerate(args) if x in opts), None)
                if idx is not None:
                    arg = args[idx+1].split()
                    if len(arg) > 1:
                        args[idx+1:idx+2] = arg

        subnamespace = super().parse_args(args)
        parsed_args_dict = vars(subnamespace)
        filtered_args = {
            key: value 
            for key, value in parsed_args_dict.items() 
        }
        return filtered_args
    
    def parse_command(self, args=None):
        command = sys.argv[1] if args is None else args.pop('command', None)
        if command not in self._subparsers._actions[-1].choices.keys():
            self.error_message("Correct program")
        
        return command

    def error_message(self, message, print_help=True):
        print(message, end=' ')
        if print_help:
            self.print_help()
        sys.exit(1)
    

class LowercaseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            values = str(values).lower()
        setattr(namespace, self.dest, values)


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



def parse_params():
    # Data
    train_parser = argparse.ArgumentParser(add_help=False)
    train_parser.add_argument('--problem', default='wcrp', help="The problem to solve")
    train_parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    train_parser.add_argument('--edge_threshold', default='0', type=str, help="How many of all possible edges to consider")
    train_parser.add_argument('--edge_method', type=str, default=None, help="Method for getting edges ('dist'|'knn')")
    train_parser.add_argument('--batch_size', type=int, default=256, help='Number of instances per batch during training')
    train_parser.add_argument('--epoch_size', type=int, default=128_000, help='Number of instances per epoch during training')
    train_parser.add_argument('--val_size', type=int, default=0, 
                            help='Number of instances used for reporting validation performance (0 to deactivate validation)')
    train_parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    train_parser.add_argument('--eval_batch_size', type=int, default=256, help="Batch size to use during (baseline) evaluation")
    train_parser.add_argument('--train_dataset', type=str, default=None, help='Name of dataset to use for training')

    # WSmart+ Route
    train_parser.add_argument('--eval_time_days', type=int, default=1, help='Number of days to perform validation (if train_time=True)')
    train_parser.add_argument('--train_time', action='store_true', help='Set to train the model over multiple days on the same graphs (n_days=n_epochs)')
    train_parser.add_argument('--area', type=str, default='riomaior', help='County area of the bins locations')
    train_parser.add_argument('--waste_type', type=str, default='plastic', help='Type of waste bins selected for the optimization problem')
    train_parser.add_argument('--focus_graph', default=None, help='Path to the file with the coordinates of the graph to focus on')
    train_parser.add_argument('--focus_size', type=int, default=0, help='Number of focus graphs to include in the training data')
    train_parser.add_argument('--eval_focus_size', type=int, default=0, help='Number of focus graphs to include in the validation data')
    train_parser.add_argument('--distance_method', type=str, default='ogd', help="Method to compute distance matrix")
    train_parser.add_argument('--dm_filepath', type=str, default=None, help="Path to the file to read/write the distance matrix from/to")
    train_parser.add_argument('--waste_filepath', type=str, default=None, help="Path to the file to read the waste fill for each day from")
    train_parser.add_argument('--vertex_method', type=str, default="mmn", help="Method to transform vertex coordinates "
                            "'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'")

    # Model
    train_parser.add_argument('--model', default='am', help="Model: 'am'|'tam'|'ddam'|'pn'")
    train_parser.add_argument('--encoder', default='gat', help="Encoder: 'gat'|gac'|'tgc'|'gcn'|'mlp'")
    train_parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    train_parser.add_argument('--hidden_dim', type=int, default=512, help='Dimension of hidden layers in Enc/Dec')
    train_parser.add_argument('--n_encode_layers', type=int, default=3, help='Number of layers in the encoder/critic network')
    train_parser.add_argument('--n_encode_sublayers', type=int, default=None, help='Number of sublayers in the encoder network')
    train_parser.add_argument('--n_predict_layers', type=int, default=None, help='Number of layers in the predictor network')
    train_parser.add_argument('--n_decode_layers', type=int, default=None, help='Number of layers in the decoder network')
    train_parser.add_argument('--temporal_horizon', type=int, default=0, help="Number of previous days/epochs to take into account in predictions")
    train_parser.add_argument('--tanh_clipping', type=float, default=10., help='Clip the parameters to within +- this value using tanh. '
                            'Set to 0 to not perform any clipping.')
    train_parser.add_argument('--normalization', default='instance', help="Normalization type: 'instance'|'layer'|'batch'|'group'|'local_response'|None")
    #train_parser.add_argument('--layernorm_bias', action='store_false', help="Disable bias for LayerNorm normalization")
    train_parser.add_argument('--learn_affine', action='store_false', help="Disable learnable affine transformation during normalization")
    train_parser.add_argument('--track_stats', action='store_true', help="Track statistics during normalization")
    train_parser.add_argument('--epsilon_alpha', type=float, default=1e-05, help="Epsilon (or alpha multiplicative, for LocalResponseNorm) value")
    train_parser.add_argument('--momentum_beta', type=float, default=0.1, help="Momentum (or beta exponential, for LocalResponseNorm) value")
    train_parser.add_argument('--lrnorm_k', type=float, default=None, help="Additive factor for LocalResponseNorm")
    train_parser.add_argument('--gnorm_groups', type=int, default=4, help="Number of groups to separate channels into for GroupNorm")
    train_parser.add_argument('--activation', default='gelu', help="Activation function: 'gelu'|'gelu_tanh'|'tanh'|'tanhshrink'|'mish'|'hardshrink'|'hardtanh'|'hardswish'|'glu'|"
                            "'relu'|'leakyrelu'|'silu'|'selu'|'elu'|'celu'|'prelu'|'rrelu'|'sigmoid'|'logsigmoid'|'hardsigmoid'|'threshold'|'softplus'|'softshrink'|'softsign'")
    #train_parser.add_argument('--af_inplace', action='store_true', help="Enable inplace operations for the activation function")
    train_parser.add_argument('--af_param', type=float, default=1.0, help="Parameter for the activation function formulation")
    train_parser.add_argument('--af_threshold', type=float, default=None, help="Threshold value for the activation function")
    train_parser.add_argument('--af_replacement', type=float, default=None, help="Replacement value for the activation function (above/below threshold)")
    train_parser.add_argument('--af_nparams', type=int, default=3, help="Number of parameters a for the Parametric ReLU (PReLU) activation")
    train_parser.add_argument('--af_urange', type=float, nargs='+', default=[0.125, 1/3], help="Range for the uniform distribution of the Randomized Leaky ReLU (RReLU) activation")
    train_parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate for the model")
    train_parser.add_argument('--aggregation_graph', default='avg', help="Graph embedding aggregation function: 'sum'|'avg'|'max'|None")
    train_parser.add_argument('--aggregation', default='sum', help="Node embedding aggregation function: 'sum'|'avg'|'max'")
    train_parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads")
    train_parser.add_argument('--mask_inner', action='store_false', help="Mask inner values during decoding")
    train_parser.add_argument('--mask_logits', action='store_false', help="Mask logits during decoding")
    train_parser.add_argument('--mask_graph', action='store_true', help="Mask next node selection (using edges) during decoding")

    # Training
    #train_parser.add_argument('--rl_algorithm', type=str, default='reinforce', help="Reinforcement Learning algorithm to train the model: 'reinforce'|'ppo'|'ppo_cl'")
    train_parser.add_argument('--n_epochs', type=int, default=25, help='The number of epochs to train')
    train_parser.add_argument('--epoch_start', type=int, default=0, help='Start at epoch # (relevant for learning rate decay)')
    train_parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    train_parser.add_argument('--lr_critic_value', type=float, default=1e-4, help="Set the learning rate for the critic/value network")
    train_parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed to use')
    train_parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum L2 norm for gradient clipping (0 to disable clipping)')
    train_parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    train_parser.add_argument('--enable_scaler', action='store_true', help='Enables CUDA scaler for automatic mixed precision training')
    train_parser.add_argument('--exp_beta', type=float, default=0.8, help='Exponential moving average baseline decay')
    train_parser.add_argument('--baseline', default=None, help="Baseline to use: 'rollout'|'critic'|'exponential'|None")
    train_parser.add_argument('--bl_alpha', type=float, default=0.05, help='Significance in the t-test for updating rollout baseline')
    train_parser.add_argument('--bl_warmup_epochs', type=int, default=-1, help='Number of epochs to warmup the baseline, ' 
                            'None means 1 for rollout (exponential used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    train_parser.add_argument('--checkpoint_encoder', action='store_true', help='Set to decrease memory usage by checkpointing encoder')
    train_parser.add_argument('--shrink_size', type=int, default=None, help='Shrink the batch size if at least this many instances in the batch'
                            ' are finished to save memory (default None means no shrinking)')
    train_parser.add_argument('--data_distribution', type=str, default=None, help='Data distribution to use during training,'
                            ' defaults and options depend on problem. "empty"|"const"|"unif"|"dist"|"gamma[1-4]|"emp""')
    train_parser.add_argument('--accumulation_steps', type=int, default=1, 
                            help='Gradient accumulation steps during training (effective batch_size = batch_size * accumulation_steps)')
    train_parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    train_parser.add_argument('--resume', help='Resume from previous checkpoint file')
    train_parser.add_argument('--post_processing_epochs', type=int, default=0, help='The number of epochs for post-processing')
    train_parser.add_argument('--lr_post_processing', type=float, default=0.001, help="Set the learning rate for post-processing")
    train_parser.add_argument('--efficiency_weight', type=float, default=0.8, help="Weight for the efficiency in post-processing")
    train_parser.add_argument('--overflow_weight', type=float, default=0.2, help="Weight for the bin overflows in post-processing")

    """# ===== Proximal Policy Optimization (PPO) =====
    train_parser.add_argument('--clip_eps', type=float, default=0.2, help='Initial policy objective clipping parameter (usually small, between 0.1 to 0.3). '
                              'How far can the new policy go from the old policy while still improving the objective function?')
    train_parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient towards overall PPO loss')
    train_parser.add_argument('--entropy_coef', type=float, default=0.01, help='Coefficient of entropy bonus (higher values favour more exploration instead of exploitation).')
    train_parser.add_argument('--k_ppo_epochs', type=int, default=4, help='Number of ppo inner epochs per batch')
    train_parser.add_argument('--mini_batch_size', type=int, default=64, help='Size of mini-batches')
    train_parser.add_argument('--disc_gamma', type=float, default=0.99, help='Discount factor (between 0 to 1)')
    train_parser.add_argument('--gae_lambda', type=float, default=0.95, help='Generalized Advantage Estimation lambda (between 0 to 1, and usually close to 1)')
    train_parser.add_argument('--normalize_adv', type='store_false', help='Whether to normalize advantage')
    train_parser.add_argument('--target_kl', type=float, default=0.015, help='What KL divergence we think is appropriate between new and old policies after an update. '
                              'This will get used for early stopping (usually small, between 0.01 to 0.05).')
    train_parser.add_argument('--separate_value_opt', action='store_true', help='Create separate optimizer for baseline')
    train_parser.add_argument('--value_updates', type=int, default=1, help='Number of value network updates (if using a separate optimizer)')

    # ===== N-Step PPO-Curriculum Learning (PPO-CL) =====
    train_parser.add_argument('--n_steps', type=int, default=25, help='The number of steps for n-step PPO')
    train_parser.add_argument('--max_T_train', type=int, default=250, help='The maximum inference T used for training')
    train_parser.add_argument('--max_T_test', type=int, default=1000, help='The maximum inference T used for test')
    train_parser.add_argument('--CL_factor', type=float, default=2.0, help='The Curriculum Learning difficulty level increase factor (higher = faster difficulty increase).')
    train_parser.add_argument('--CL_best', type=bool, default=False, help='Whether use the best solution from the PPO-CL rollout')
    """

    # Optimizer and learning rate scheduler
    train_parser.add_argument('--optimizer', type=str, default='rmsprop', 
                            help="Optimizer: 'adam'|'adamax'|'adamw'|'radam'|'nadam'|'sadam'|'adadelta'|'adagrad'|'rmsprop'|'rprop'|'lbfgs'|'asgd'|'sgd'")
    train_parser.add_argument('--lr_scheduler', type=str, default='lambda', 
                            help="Learning rate scheduler: 'exp'|'step'|'mult'|'lambda'|'const'|'poly'|'multistep'|'cosan'|'linear'|'cosanwr'|'plateau'")
    train_parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    train_parser.add_argument('--lr_min_value', type=float, default=0.0, help='Minimum learning rate for CosineAnnealingLR|CosineAnnealingWarmRestarts|ReduceLROnPlateau')
    train_parser.add_argument('--lr_min_decay', type=float, default=1e-8, help='Minimum decay applied to learning rate for LinearLR|ReduceLROnPlateau')
    train_parser.add_argument('--lrs_step_size', type=int, default=1, help='Period of learning rate decay for StepLR')
    train_parser.add_argument('--lrs_total_steps', type=int, default=5, help='Number of steps that the scheduler updates the lr for ConstantLR|LinearLR|PolynomialLR')
    train_parser.add_argument('--lrs_restart_steps', type=int, default=7, help='Number of steps until the first restart for CosineAnnealingWarmRestarts')
    train_parser.add_argument('--lrs_rfactor', type=int, default=2, help='A factor that, after a restart, increases the steps for the next restart for CosineAnnealingWarmRestarts.')
    train_parser.add_argument('--lrs_milestones', type=int, nargs='+', default=[7, 14, 21, 28], help='List of epoch indices (must be increasing) for MultiStepLR.')
    train_parser.add_argument('--lrs_mode', type=str, default='min', help='Scheduler mode for ReduceLROnPlateau')
    train_parser.add_argument('--lrs_dfactor', type=float, default=0.1, help='A factor by which the learning rate will be decreased for ReduceLROnPlateau')
    train_parser.add_argument('--lrs_patience', type=int, default=10, help='Number of epochs with no improvement after which the learning rate will be updated for ReduceLROnPlateau')
    train_parser.add_argument('--lrs_thresh', type=float, default=1e-4, help='Threshold for measuring the new optimum, to only focus on significant changes for ReduceLROnPlateau')
    train_parser.add_argument('--lrs_thresh_mode', type=str, default='rel', choices=['rel', 'abs'], help='Scheduler threshold mode for ReduceLROnPlateau')
    train_parser.add_argument('--lrs_cooldown', type=int, default=0, help='Number of epochs to wait before resuming normal operation after lr has been reduced for ReduceLROnPlateau')

    # Cost function weights
    train_parser.add_argument('--w_waste', type=float, default=None, help="Weight for the waste collected")
    train_parser.add_argument('--w_length', '--w_len', type=float, default=None, help="Weight for the route length")
    train_parser.add_argument('--w_overflows', '--w_over', type=float, default=None, help="Weight for the number of overflows")
    train_parser.add_argument('--w_lost', type=float, default=None, help="Weight for the amount of waste lost when bins overflow")
    train_parser.add_argument('--w_penalty', '--w_pen', type=float, default=None, help="Weight for the penalty")
    train_parser.add_argument('--w_prize', type=float, default=None, help="Weight for the prize")

    # Output
    train_parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    train_parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    train_parser.add_argument('--run_name', default=None, help='Name to identify the run')
    train_parser.add_argument('--output_dir', default='model_weights', help='Directory to write output models to')
    train_parser.add_argument('--checkpoint_epochs', type=int, default=1, help='Save checkpoint every n epochs, 0 to save no checkpoints')
    train_parser.add_argument('--wandb_mode', default='offline', help="Weights and biases mode: 'online'|'offline'|'disabled'|None")
    train_parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    train_parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    # Meta-Reinforcement Learning (MRL)
    mrl_train_parser = argparse.ArgumentParser(parents=[train_parser], add_help=False)
    mrl_train_parser.add_argument('--mrl_method', type=str, default='cb', choices=['tdl', 'rwa', 'cb', 'morl'], help='Method to use for Meta-Reinforcement Learning')
    mrl_train_parser.add_argument('--mrl_history', type=int, default=10, help="Number of previous days/epochs to take into account during Meta-Reinforcement Learning")
    mrl_train_parser.add_argument('--mrl_range', type=float, nargs='+', default=[0.01, 5.0], help="Maximum and minimum values for Meta-Reinforcement Learning with dynamic hyperparameters")
    mrl_train_parser.add_argument('--mrl_exploration_factor', type=float, default=2.0, help="Factor that controls the exploration vs. exploitation balance")
    mrl_train_parser.add_argument('--mrl_lr', type=float, default=1e-3, help="Set the learning rate for Meta-Reinforcement Learning")
    mrl_train_parser.add_argument('--tdl_lr_decay', type=float, default=1.0, help='Learning rate decay for Temporal Difference Learning')
    mrl_train_parser.add_argument('--cb_exploration_method', type=str, default='ucb', help="Method for exploration in Contextual Bandits: 'ucb'|'thompson_sampling'|'epsilon_greedy'")
    mrl_train_parser.add_argument('--cb_num_configs', type=int, default=10, help="Number of weight configurations to generate in Contextual Bandits")
    mrl_train_parser.add_argument('--cb_context_features', type=str, nargs='+', default=['waste', 'overflow', 'length', 'visited_ratio', 'day'], help="Features for Contextual Bandits")
    mrl_train_parser.add_argument('--cb_features_aggregation', default='avg', help="Context features aggregation function in Contextual Bandits: 'sum'|'avg'|'max'")
    mrl_train_parser.add_argument('--cb_epsilon_decay', type=float, default=0.995, help="Decay factor for epsilon (=exploration_factor in 'epsilon_greedy')")
    mrl_train_parser.add_argument('--cb_min_epsilon', type=float, default=0.01, help="Minimum value for epsilon (=exploration_factor in 'epsilon_greedy')")
    mrl_train_parser.add_argument('--morl_objectives', type=str, nargs='+', default=['waste_efficiency', 'overflow_rate'], help="Objectives for Multi-Objective RL")
    mrl_train_parser.add_argument('--morl_adaptation_rate', type=float, default=0.1, help="Adaptation rate in Multi-Objective RL")
    mrl_train_parser.add_argument('--rwa_model', type=str, default='rnn', choices=['rnn'], help='Neural network to use for Reward Weight Adjustment')
    mrl_train_parser.add_argument('--rwa_optimizer', type=str, default='rmsprop', help="Optimizer: 'adamax'|'adam'|'adamw'|'radam'|'nadam'|'rmsprop'")
    mrl_train_parser.add_argument('--rwa_embedding_dim', type=int, default=128, help='Dimension of input embedding for Reward Weight Adjustment model')
    mrl_train_parser.add_argument('--rwa_batch_size', type=int, default=256, help="Batch size to use for Reward Weight Adjustment model")
    mrl_train_parser.add_argument('--rwa_step', type=int, default=100, help='Update Reward Weight Adjustment model every rwa_step steps')
    mrl_train_parser.add_argument('--rwa_update_step', type=int, default=100, help='Update Reward Weight Adjustment weights every rwa_update_step steps')
    #mrl_train_parser.add_argument('--val_window', type=int, default=5, help="Number of days to track for trend analysis")
    #mrl_train_parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")

    # Hyper-Parameter Optimization (HPO)
    hp_optim_parser = argparse.ArgumentParser(parents=[train_parser], add_help=False)
    hp_optim_parser.add_argument('--hop_method', type=str, default='dehbo', choices=['dea', 'bo', 'hbo', 'rs', 'gs', 'dehbo', 'pbba', 'bgpbt'],
                                help='Method to use for hyperparameter optimization')
    hp_optim_parser.add_argument('--hop_range', type=float, nargs='+', default=[0.0, 2.0], help="Maximum and minimum values for hyperparameter optimization")
    hp_optim_parser.add_argument('--hop_epochs', type=int, default=7, help='The number of epochs to optimize hyperparameters')
    #hp_optim_parser.add_argument('--val_epochs', type=int, default=7, help='The number of epochs to perform validation')
    hp_optim_parser.add_argument('--metric', type=str, default="val_loss", choices=['loss', 'val_loss', 'mean_reward', 'mae', 'mse', 'rmse', 'episode_reward_mean', 
                                'kg/km', 'overflows', 'both'], help='Metric to optimize')

    # ===== Bayesian Optimization (BO) =====
    hp_optim_parser.add_argument('--n_trials', type=int, default=20, help='Number of trials for Optuna optimization')
    hp_optim_parser.add_argument('--timeout', type=int, default=None, help='Timeout for Optuna optimization (in seconds)')
    hp_optim_parser.add_argument('--n_startup_trials', type=int, default=5, help='Number of trials to run before pruning starts')
    hp_optim_parser.add_argument('--n_warmup_steps', type=int, default=3, help='Number of epochs to wait before pruning can happen in each trial')
    hp_optim_parser.add_argument('--interval_steps', type=int, default=1, help='Pruning is evaluated every this many epochs')

    # ===== Distributed Evolutionary Algorithm (DEA) =====
    hp_optim_parser.add_argument('--eta', type=float, default=10.0, help="Controls the spread of the genetic mutations (higher = slower changes)")
    hp_optim_parser.add_argument('--indpb', type=float, default=0.2, help="Probability of mutating each gene of an individual in the population")
    hp_optim_parser.add_argument('--tournsize', type=int, default=3, help="Number of individuals to fight to be selected for reproduction")
    hp_optim_parser.add_argument('--cxpb', type=float, default=0.7, help="Probability of crossover between two parents (higher = faster convergence + less diversity)")
    hp_optim_parser.add_argument('--mutpb', type=float, default=0.2, help="Probability of an individual being mutated after crossover")
    hp_optim_parser.add_argument('--n_pop', type=int, default=20, help="Starting population for evolutionary algorithms")
    hp_optim_parser.add_argument('--n_gen', type=int, default=10, help="Number of generations to evolve")

    # ===== Differential Evolutionary Hyperband Optimization (DEHBO) =====
    hp_optim_parser.add_argument('--fevals', type=int, default=100, help='Number of function evaluations')

    # Ray Tune framework hyperparameters
    hp_optim_parser.add_argument('--cpu_cores', type=int, default=1, help="Number of CPU cores to use for hyperparameter optimization (0 uses all available cores)")
    hp_optim_parser.add_argument('--verbose', type=int, default=2, help='Verbosity level for Hyperband and Random Search (0-3)')
    hp_optim_parser.add_argument('--train_best', action='store_true', default=True, help='Train final model with best hyperparameters')
    hp_optim_parser.add_argument('--local_mode', action='store_true', help='Run ray in local mode (for debugging)')
    hp_optim_parser.add_argument('--num_samples', type=int, default=20, help='Number of times to sample from the hyperparameter space')

    # ===== Hyperband Optimization (HBO) - Ray Tune =====
    hp_optim_parser.add_argument('--max_tres', type=int, default=14, help='Maximum resources (timesteps) per trial')
    hp_optim_parser.add_argument('--reduction_factor', type=int, default=3, help='Reduction factor for Hyperband')

    # ===== Random Search (RS) - Ray Tune =====
    hp_optim_parser.add_argument('--max_failures', type=int, default=3, help='Maximum trial failures before stopping')

    # ===== Grid Search (GS) - Ray Tune =====
    hp_optim_parser.add_argument('--grid', type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5, 2.0], help='Hyperparameter values to try in grid search')
    hp_optim_parser.add_argument('--max_conc', type=int, default=4, help='Maximum number of concurrent trials for Ray Tune')

    # ===== Population Based Bandits Algorithm (PBBA) =====
    """hp_optim_parser.add_argument("--freq", type=int, default=50_000, help="How often (w.r.t. the time_attr criteria) hyperparams are perturbed/mutated")
    hp_optim_parser.add_argument("--horizon", type=int, default=1000, help="Maximum number of steps in an episode for the environment")
    hp_optim_parser.add_argument("--perturb", type=float, default=0.25, help="Fraction of hyperparams that are perturbed/mutated")
    hp_optim_parser.add_argument("--criteria", type=str, default="timesteps_total", choices=['timesteps_total', 'training_iteration'],
                                help="Criteria used to track progress of optimization process")
    hp_optim_parser.add_argument("--method", type=str, default="pb2", help="Scheduler for the optimization process: 'pbt'|'pb2'|'asha'")
    hp_optim_parser.add_argument("--algo", type=str, default="reinforce", help="Selected policy gradient algorithm: 'reinforce'|'ppo'|'impala'")"""

    # ===== Bayesian Generational Population Based Training (PGPBT) =====
    """hp_optim_parser.add_argument('--search_mode', type=str, default='hpo', choices=['hpo', 'both'], help="Type of search/optimization to perform")
    hp_optim_parser.add_argument('--num_init', type=int, default=24, help="Number of randomly initialising points")
    hp_optim_parser.add_argument('--t_ready', type=int, default=1_000_000, help="How many steps between explore/exploit")
    hp_optim_parser.add_argument('--t_ready_end', type=int, default=None, help="How many steps between explore/exploit at the end")
    hp_optim_parser.add_argument('--t_distil', type=int, default=30_000_000, help="How many steps for each distillation")
    hp_optim_parser.add_argument('--distill_every', type=int, default=40_000_000, help="Maximum timestep before distillation is triggered. " \
                                "Note that this may happen earlier if the training halts, as determined by the patience parameter")
    hp_optim_parser.add_argument('--patience', type=int, default=20, help='Number of training steps without improvements before distillation is triggered')
    hp_optim_parser.add_argument('--max_distillation', type=int, default=2, help="Maximum number of distillations/generations")
    hp_optim_parser.add_argument('--quantile_fraction', type=float, default=0.125, help="The bottom fraction of agents to be replaced at each iteration")
    hp_optim_parser.add_argument('--existing_policy', type=str,choices=['overwrite', 'resume'], default='resume', help="What to do whe log dir has existing files")
    hp_optim_parser.add_argument('--arch_policy', type=str, choices=['random', 'search'], default='search', help="How to generate the initial population after a distillation")"""

    # Generate data
    gen_data_parser = argparse.ArgumentParser(add_help=False)
    gen_data_parser.add_argument("--name", type=str, help="Name to identify dataset")
    gen_data_parser.add_argument("--filename", default=None, help="Filename of the dataset to create (ignores datadir)")
    gen_data_parser.add_argument("--data_dir", default='datasets', help="Create datasets in data")
    gen_data_parser.add_argument("--problem", type=str, default='all', 
                                help="Problem: 'tsp'|'vrp'|'pctsp'|'vrpp'|'wcrp'|'op_const'|'op_unif'|'op_dist'|'pdp'|'all'")
    gen_data_parser.add_argument("--is_gaussian", type=int, default=0)
    gen_data_parser.add_argument('--data_distributions', nargs='+', default=['all'], help="Distributions to generate for problems")
    gen_data_parser.add_argument("--dataset_size", type=int, default=128_000, help="Size of the dataset")
    gen_data_parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100], help="Sizes of problem instances")
    gen_data_parser.add_argument('--sigma', type=float, nargs='+', default=0.6)
    gen_data_parser.add_argument('--penalty_factor', type=float, default=3.0, help="Penalty factor for problems")
    gen_data_parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    gen_data_parser.add_argument('--seed', type=int, default=42, help="Random seed")
    gen_data_parser.add_argument('--n_epochs', type=int, default=1, help='The number of epochs to generate data for')
    gen_data_parser.add_argument('--epoch_start', type=int, default=0, help='Start at epoch #')
    gen_data_parser.add_argument('--dataset_type', type=str, choices=['train', 'train_time', 'test_simulator'], help='Set type of dataset to generate')
    gen_data_parser.add_argument('--area', type=str, default='riomaior', help='County area of the bins locations')
    gen_data_parser.add_argument('--waste_type', type=str, default='plastic', help='Type of waste bins selected for the optimization problem')
    gen_data_parser.add_argument('--focus_graphs', nargs='+', default=None, help='Path to the files with the coordinates of the graphs to focus on')
    gen_data_parser.add_argument('--focus_size', type=int, default=0, help='Number of focus graphs to include in the data')
    gen_data_parser.add_argument('--vertex_method', type=str, default="mmn", help="Method to transform vertex coordinates "
                                "'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'")

    # Evaluate algorithms
    eval_parser = argparse.ArgumentParser(add_help=False)
    eval_parser.add_argument("--datasets", type=str, nargs='+', help="Filename of the dataset(s) to evaluate")
    eval_parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    eval_parser.add_argument("-o", default=None, help="Name of the results file to write")
    eval_parser.add_argument('--val_size', type=int, default=12_800, help='Number of instances used for reporting validation performance')
    eval_parser.add_argument('--offset', type=int, default=0, help='Offset where to start in dataset')
    eval_parser.add_argument('--eval_batch_size', type=int, default=256, help="Batch size to use during (baseline) evaluation")
    eval_parser.add_argument('--decode_type', type=str, default='greedy', help='Decode type, greedy or sampling')
    eval_parser.add_argument('--width', type=int, nargs='+', help='Sizes of beam to use for beam search (or number of samples for sampling), '
                            '0 to disable (default), -1 for infinite')
    eval_parser.add_argument('--decode_strategy', type=str, help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    eval_parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1, help="Softmax temperature (sampling or bs)")
    eval_parser.add_argument('--model', type=str)
    eval_parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    eval_parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    eval_parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    eval_parser.add_argument('--max_calc_batch_size', type=int, default=12_800, help='Size for subbatches')
    eval_parser.add_argument('--results_dir', default='results', help="Name of evaluation results directory")
    eval_parser.add_argument('--multiprocessing', action='store_true', help='Use multiprocessing to parallelize over multiple GPUs')
    eval_parser.add_argument('--graph_size', type=int, default=50, help="The size of the problem graph")
    eval_parser.add_argument('--area', type=str, default='riomaior', help='County area of the bins locations')
    eval_parser.add_argument('--waste_type', type=str, default='plastic', help='Type of waste bins selected for the optimization problem')
    eval_parser.add_argument('--focus_graph', default=None, help='Path to the file with the coordinates of the graph to focus on')
    eval_parser.add_argument('--focus_size', type=int, default=0, help='Number of focus graphs to include in the training data')
    eval_parser.add_argument('--edge_threshold', default='0', type=str, help="How many of all possible edges to consider")
    eval_parser.add_argument('--edge_method', type=str, default=None, help="Method for getting edges ('dist'|'knn')")
    eval_parser.add_argument('--distance_method', type=str, default='ogd', help="Method to compute distance matrix")
    eval_parser.add_argument('--vertex_method', type=str, default="mmn", help="Method to transform vertex coordinates "
                            "'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'")

    # Evaluate models and policies on simulator
    test_sim_parser = argparse.ArgumentParser(add_help=False)
    test_sim_parser.add_argument('--policies', type=str, nargs='+', help="Name of the policy(ies) to test on the WSR simulator")
    test_sim_parser.add_argument('--data_distribution', '--dd', type=str, default='gamma1', help="Distribution to generate the bins daily waste fill")
    test_sim_parser.add_argument('--problem', default='vrpp', help="The problem the model was trained to solve")
    test_sim_parser.add_argument('--size', type=int, default=50, help="The size of the problem graph")
    test_sim_parser.add_argument('--days', type=int, default=31, help="Number of days to run the simulation for")
    test_sim_parser.add_argument('--seed', type=int, default=42, help="Random seed")
    test_sim_parser.add_argument('--output_dir', default='output', help="Name of WSR simulator test output directory")
    test_sim_parser.add_argument('--checkpoint_dir', default='temp', help="Name of WSR simulator test runs checkpoint directory")
    test_sim_parser.add_argument('--checkpoint_days', '--cpd', type=int, default=5, help='Save checkpoint every n days, 0 to save no checkpoints')
    test_sim_parser.add_argument('--n_samples', type=int, default=1, help="Number of simulation samplings for each policy")
    test_sim_parser.add_argument('--resume', action='store_true', help='Resume testing (relevant for saving results)')
    test_sim_parser.add_argument('--pregular_level', '--lvl', type=int, nargs='+', help="Regular policy level")
    test_sim_parser.add_argument('--plastminute_cf', '--cf', type=int, nargs='+', help="CF value for last minute/last minute and path policies")
    test_sim_parser.add_argument('--lookahead_configs', '--lac', type=str, nargs='+', help="Parameter configuration for policy Look-Ahead and variants")
    test_sim_parser.add_argument('--gurobi_param', '--gp', type=float, default=0.84, nargs='+', help='Param value for Gurobi VRPP policy '
                            '(higher = more conservative with regards to amount of overflows)')
    test_sim_parser.add_argument('--hexaly_param', '--hp', type=float, default=2.0, nargs='+', help='Param value for Hexaly optimizer policy '
                            '(higher = more conservative with regards to amount of overflows)')
    test_sim_parser.add_argument('--cpu_cores', '--cc', type=int, default=0, help="Number of max CPU cores to use (0 uses all available cores)")
    test_sim_parser.add_argument('--n_vehicles', type=int, default=1, help="Number of vehicles")
    test_sim_parser.add_argument('--area', type=str, default='riomaior', help='County area of the bins locations')
    test_sim_parser.add_argument('--waste_type', type=str, default='plastic', help='Type of waste bins selected for the optimization problem')
    test_sim_parser.add_argument('--bin_idx_file', type=str, default=None, help="File with the indices of the bins to use in the simulation")
    test_sim_parser.add_argument('--decode_type', '--dt', type=str, default='greedy', help='Decode type, greedy or sampling')
    test_sim_parser.add_argument('--temperature', type=parse_softmax_temperature, default=1, help="Softmax temperature (sampling or bs)")
    test_sim_parser.add_argument('--edge_threshold', '--et', default='0', type=str, help='How many of all possible edges to consider')
    test_sim_parser.add_argument('--edge_method', '--em', type=str, default=None, help="Method for getting edges ('dist'|'knn')")
    test_sim_parser.add_argument('--vertex_method', '--vm', type=str, default="mmn", help="Method to transform vertex coordinates "
                            "'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'")
    test_sim_parser.add_argument('--distance_method', '--dm', type=str, default='ogd', help="Method to compute distance matrix")
    test_sim_parser.add_argument('--dm_filepath', '--dmf', type=str, default=None, help="Path to the file to read/write the distance matrix from/to")
    test_sim_parser.add_argument('--waste_filepath', type=str, default=None, help="Path to the file to read the waste fill for each day from")
    test_sim_parser.add_argument('--run_tsp', action='store_true', help="Activate fast_tsp for all policies.")
    test_sim_parser.add_argument('--cache_regular', action='store_false', help="Deactivate caching for policy regular.")
    test_sim_parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    test_sim_parser.add_argument('--server_run', action='store_true', help="Simulation will be executed in a remote server")
    test_sim_parser.add_argument('--env_file', type=str, default='vars.env', help="Name of the file that contains the environment variables")
    test_sim_parser.add_argument('--gplic_file', type=str, default=None, help="Name of the file that contains the license to use for Gurobi")
    test_sim_parser.add_argument('--hexlic_file', type=str, default=None, help="Name of the file that contains the license to use for Gurobi")
    test_sim_parser.add_argument('--symkey_name', type=str, default=None, help="Name of the cryptographic key used to access the API keys")
    test_sim_parser.add_argument('--gapik_file', type=str, default=None, help="Name of the file that contains the key to use for the Google API")

    # CRUD (Create, Read, Update, and Delete) and Criptographic functionality for file system entries
    files_parser = argparse.ArgumentParser(add_help=False)
    files_subparsers = files_parser.add_subparsers(help="file system command", dest="fs_command")

    # Update file system entries
    update_parser = files_subparsers.add_parser('update', help='Update file system entries')
    update_parser.add_argument('--target_entry', type=str, help="Path to the file to the file system entry we want to update")
    update_parser.add_argument('--output_key', type=str, default=None, help="Key of the values we want to update in the files")
    update_parser.add_argument('--filename_pattern', type=str, default=None, help="Pattern to match names of files to update (target_entry must be directory)")
    update_parser.add_argument('--update_operation', type=str, default=None, action=UpdateFunctionMapActionFactory(inplace=True), help="Operation to update the file values")
    update_parser.add_argument('--update_value', type=float, default=0.0, help="Value for the update operation")
    update_parser.add_argument('--update_preview', action='store_true', help="Preview how files/directories will look like after the update")
    update_parser.add_argument('--stats_function', type=str, default=None, action=UpdateFunctionMapActionFactory(), help="Function to perform over the file values")
    update_parser.add_argument('--output_filename', type=str, default=None, help="Name of the file we want to save the output values to")
    update_parser.add_argument('--input_keys', type=str, default=(None, None), nargs='*', help="Key(s) of the values we want to use as input to update the other key in the files")

    # Delete file system entries
    delete_parser = files_subparsers.add_parser('delete', help='Delete file system entries')
    delete_parser.add_argument('--wandb', action="store_false", help="Flag to delete the train wandb log directory")
    delete_parser.add_argument('--log_dir', default='logs', help='Directory of train logs')
    delete_parser.add_argument('--log', action="store_false", help="Flag to delete the train log directory")
    delete_parser.add_argument('--output_dir', default='model_weights', help='Directory to write output models to')
    delete_parser.add_argument('--output', action="store_false", help="Flag to delete the train output models directory")
    delete_parser.add_argument("--data_dir", default='datasets', help="Directory of generated datasets")
    delete_parser.add_argument('--data', action="store_true", help="Flag to delete the datasets directory")
    delete_parser.add_argument('--eval_dir', default='results', help="Name of the evaluation results directory")
    delete_parser.add_argument('--eval', action="store_true", help="Flag to delete the evaluation results directory")
    delete_parser.add_argument('--test_dir', default='output', help="Name of the WSR simulator test output directory")
    delete_parser.add_argument('--test', action="store_true", help="Flag to delete the WSR simulator test output directory")
    delete_parser.add_argument('--test_checkpoint_dir', default='temp', help="Name of WSR simulator test runs checkpoint directory")
    delete_parser.add_argument('--test_checkpoint', action="store_true", help="Flag to delete the WSR simulator test runs checkpoint directory")
    delete_parser.add_argument('--cache', action="store_true", help="Flag to delete the cache directories")
    delete_parser.add_argument('--delete_preview', action='store_true', help="Preview which files/directories will be removed")

    # Cryptography
    crypto_parser = files_subparsers.add_parser('cryptography', help='Perform cryptographic operations on file system entries')
    crypto_parser.add_argument('--symkey_name', type=str, default=None, help="Name of the key for the files to save the salt and key/hash parameters to")
    crypto_parser.add_argument('--env_file', type=str, default='vars.env', help="Name of the file that contains the environment variables")
    crypto_parser.add_argument('--salt_size', type=int, default=16)
    crypto_parser.add_argument('--key_length', type=int, default=32)
    crypto_parser.add_argument('--hash_iterations', type=int, default=100_000)
    crypto_parser.add_argument('--input_path', type=str, default=None)
    crypto_parser.add_argument('--output_path', type=str, default=None)

    # Graphical User Interface (GUI)
    gui_parser = argparse.ArgumentParser(add_help=False)
    gui_parser.add_argument('--app_style', action=LowercaseAction, type=str, default="fusion", help="Style for the GUI application")
    gui_parser.add_argument('--test_only', action='store_true', help="Test mode for the GUI (commands are only printed, not executed).")

    # Test suite
    test_suite_parser = argparse.ArgumentParser(
        add_help=False, epilog=__doc__,
        description='Test runner for arg_parser test suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Test selection
    test_suite_parser.add_argument('-m', '--module', nargs='+', choices=list(TEST_MODULES.keys()), help='Specific test module(s) to run')
    test_suite_parser.add_argument('-c', '--class', dest='test_class', help='Specific test class to run (e.g., TestTrainCommand)')
    test_suite_parser.add_argument('-t', '--test', dest='test_method', help='Specific test method to run (e.g., test_train_default_parameters)')
    test_suite_parser.add_argument('-k', '--keyword', help='Run tests matching the given keyword expression')
    test_suite_parser.add_argument('--markers', help='Run tests matching the given marker expression')
    
    # Test execution options
    test_suite_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    test_suite_parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    test_suite_parser.add_argument('--ff', '--failed-first', dest='failed_first', action='store_true', help='Run failed tests first')
    test_suite_parser.add_argument('-x', '--exitfirst', dest='maxfail', action='store_const', const=1, help='Exit on first failure')
    test_suite_parser.add_argument('--maxfail', type=int, help='Exit after N failures')
    test_suite_parser.add_argument('--tb', choices=['auto', 'long', 'short', 'line', 'native', 'no'], default='auto', help='Traceback print mode')
    test_suite_parser.add_argument('--capture', choices=['auto', 'no', 'sys', 'fd'], default='auto', help='Capture mode for output')
    test_suite_parser.add_argument('-n', '--parallel', action='store_true', help='Run tests in parallel (requires pytest-xdist)')
    
    # Information commands
    test_suite_parser.add_argument('-l', '--list', action='store_true', help='List all available test modules')
    test_suite_parser.add_argument('--list-tests', action='store_true', help='List all tests in specified module(s) or all tests')
    test_suite_parser.add_argument('--test-dir', default='tests', help='Directory containing test files (default: tests)')

    # Main parser
    parser = ConfigsParser(description="Neural Combinatorial Optimization with RL")
    subparsers = parser.add_subparsers(help="command", dest="command", required=True)

    # Add subparsers
    train_subparser = subparsers.add_parser('train', parents=[train_parser], add_help=False)
    mrl_train_subparser = subparsers.add_parser('mrl_train', parents=[mrl_train_parser], add_help=False)
    hp_optim_subparser = subparsers.add_parser('hp_optim', parents=[hp_optim_parser], add_help=False)
    gen_data_subparser = subparsers.add_parser('gen_data', parents=[gen_data_parser], add_help=False)
    eval_subparser = subparsers.add_parser('eval', parents=[eval_parser], add_help=False)
    test_sim_subparser = subparsers.add_parser('test_sim', parents=[test_sim_parser], add_help=False)
    files_subparser = subparsers.add_parser('file_system', parents=[files_parser], add_help=False)
    test_suite_subparser = subparsers.add_parser('test_suite', parents=[test_suite_parser], add_help=False)
    gui_subparser = subparsers.add_parser('gui', parents=[gui_parser], add_help=False)

    args = parser.parse_process_args()
    command = parser.parse_command(args)
    if command is None:
        parser.print_help()
        sys.exit(1)
    elif command in ['train', 'hp_optim', 'mrl_train']:
        assert args['epoch_size'] % args['batch_size'] == 0, "Epoch size must be integer multiple of batch size!"
        if args['bl_warmup_epochs'] < 0:
            args['bl_warmup_epochs'] = 1 if 'baseline' in args and args['baseline'] == 'rollout' else 0

        assert (args['bl_warmup_epochs'] == 0) or (args['baseline'] == 'rollout')
        if args['encoder'] in SUB_NET_ENCS and args['n_encode_sublayers'] is None:
            args['n_encode_sublayers'] = args['n_encode_layers']
        
        assert args['encoder'] not in SUB_NET_ENCS or args['n_encode_sublayers'] > 0, \
        f"Must select a positive integer for 'n_encode_sublayers' arg for {args['encoder']} encoder"
        if args['model'] in PRED_ENC_MODELS and args['n_predict_layers'] is None:
            args['n_predict_layers'] = args['n_encode_layers']

        assert args['model'] not in PRED_ENC_MODELS or args['n_predict_layers'] > 0, \
        f"Must select a positive integer for 'n_predict_layers' arg for {args['model']} model" 
        if args['model'] in ENC_DEC_MODELS and args['n_decode_layers'] is None:
            args['n_decode_layers'] = args['n_encode_layers']

        assert args['model'] not in ENC_DEC_MODELS or args['n_decode_layers'] > 0, \
        f"Must select a positive integer for 'n_decode_layers' arg for {args['model']} model" 
        if 'run_name' in args and args['run_name'] is not None:
            args['run_name'] = "{}_{}".format(args['run_name'], time.strftime("%Y%m%dT%H%M%S"))
        else:
            args['run_name'] = "{}{}{}{}_{}".format(
                args['model'], args['encoder'], 
                args['temporal_horizon'] if args['temporal_horizon'] > 0 else "",
                "_{}".format(args['data_distribution']) if 'data_distribution' in args and args['data_distribution'] is not None else "", 
                time.strftime("%Y%m%dT%H%M%S"))
        
        args['save_dir'] = os.path.join(
            args['output_dir'],
            "{}_{}".format(args['problem'], args['graph_size']),
            args['run_name']
        )
        if 'area' in args and args['area'] is not None:
            args['area'] = re.sub(r'[^a-zA-Z]', '', args['area'].lower())
            #args['area'] = args['area'].translate(str.maketrans('', '', '-_ ')).lower()
            assert args['area'] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(args['area'], MAP_DEPOTS.keys())

        if 'waste_type' in args and args['waste_type'] is not None:
            assert 'area' in args and args['area'] is not None
            args['waste_type'] = re.sub(r'[^a-zA-Z]', '', args['waste_type'].lower())
            assert args['waste_type'] in WASTE_TYPES.keys() or args['waste_type'] is None, \
            "Unknown waste type {}, available waste types: {}".format(args['waste_type'], WASTE_TYPES.keys())

        args['edge_threshold'] = float(args['edge_threshold']) if '.' in args['edge_threshold'] else int(args['edge_threshold'])
        if command == 'hp_optim':
            assert args['cpu_cores'] >= 0, "Number of CPU cores must be non-negative integer"
            assert args['cpu_cores'] <= cpu_count(), "Number of CPU cores to use cannot exceed system specifications"
            if args['cpu_cores'] == 0:
                args['cpu_cores'] = cpu_count()
        return command, args
    
    if command == 'gen_data':
        assert 'filename' not in args or args['filename'] is None or (
            len(args['problem']) == 1 and len(args['graph_sizes']) == 1
            ), "Can only specify filename when generating a single dataset"
        
        assert 'focus_graphs' not in args or args['focus_graphs'] is None or len(args['focus_graphs']) == len(args['graph_sizes'])
        if 'focus_graphs' not in args or args['focus_graphs'] is None: 
            args['focus_graphs'] = [None] * len(args['graph_sizes'])
        else:
            args['area'] = re.sub(r'[^a-zA-Z]', '', args['area'].lower())
            assert args['area'] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(args['area'], MAP_DEPOTS.keys())

        args['waste_type'] = re.sub(r'[^a-zA-Z]', '', args['waste_type'].lower())
        assert args['waste_type'] in WASTE_TYPES.keys() or args['waste_type'] is None, \
        "Unknown waste type {}, available waste types: {}".format(args['waste_type'], WASTE_TYPES.keys())
        return command, args
    
    if command == 'eval':
        assert 'o' not in args or args['o'] is None or (
            len(args['datasets']) == 1 and len(args['width']) <= 1
            ), "Cannot specify result filename with more than one dataset or more than one width"
        
        args['area'] = re.sub(r'[^a-zA-Z]', '', args['area'].lower())
        assert args['area'] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(args['area'], MAP_DEPOTS.keys())

        args['waste_type'] = re.sub(r'[^a-zA-Z]', '', args['waste_type'].lower())
        assert args['waste_type'] in WASTE_TYPES.keys() or args['waste_type'] is None, \
        "Unknown waste type {}, available waste types: {}".format(args['waste_type'], WASTE_TYPES.keys())
        return command, args
    
    if command == 'test_sim':
        assert args['days'] >= 1, "Must run the simulation for 1 or more days"
        assert args['n_samples'] > 0, "Number of samples must be non-negative integer"

        args['area'] = re.sub(r'[^a-zA-Z]', '', args['area'].lower())
        assert args['area'] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(args['area'], MAP_DEPOTS.keys())

        args['waste_type'] = re.sub(r'[^a-zA-Z]', '', args['waste_type'].lower())
        assert args['waste_type'] in WASTE_TYPES.keys() or args['waste_type'] is None, \
        "Unknown waste type {}, available waste types: {}".format(args['waste_type'], WASTE_TYPES.keys())

        args['edge_threshold'] = float(args['edge_threshold']) if '.' in args['edge_threshold'] else int(args['edge_threshold'])
        assert args['cpu_cores'] >= 0, "Number of CPU cores must be non-negative integer"
        assert args['cpu_cores'] <= cpu_count(), "Number of CPU cores to use cannot exceed system specifications"
        if args['cpu_cores'] == 0:
            args['cpu_cores'] = cpu_count()
        return command, args

    if command == 'file_system':
        fs_comm = args.pop('fs_command')
        if fs_comm not in FS_COMMANDS:
            raise argparse.ArgumentError("ERROR: unknown File System (inner) command " + fs_comm)

        assert not ('stats_function' in args and args['stats_function'] is not None) or \
            not ('update_operation' in args and args['update_operation'] is not None), \
            "'update_operation' and 'stats_function' arguments are mutually exclusive"
        return (command, fs_comm), args
    
    if command == 'gui':
        assert args['app_style'] in [None] + APP_STYLES, \
        f"Invalid application style '{args['app_style']}' - app_style value must be: {[None] + APP_STYLES}"
        return command, args
    
    if command == 'test_suite':
        return command, args

    raise argparse.ArgumentError("ERROR: unknown command " + args['command'])