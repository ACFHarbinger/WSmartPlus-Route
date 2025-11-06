import os
import sys
import json
import torch
import wandb
import random
import traceback
import numpy as np

from tensorboard_logger import Logger as TbLogger
from backend.src.utils.arg_parser import parse_params
from backend.src.utils.functions import load_data, load_problem
from backend.src.utils.definitions import (
    ROOT_DIR, HOP_KEYS,
    update_lock_wait_time,
)
from backend.src.utils.setup_utils import (
    setup_cost_weights, 
    setup_model_and_baseline, 
    setup_optimizer_and_lr_scheduler
)
from .reinforcement_learning.epoch import validate
from .reinforcement_learning.reinforce import (
    train_reinforce_epoch, train_reinforce_over_time, 
    train_reinforce_over_time_cb, train_reinforce_over_time_tdl, 
    train_reinforce_over_time_morl, train_reinforce_over_time_rwa
)
from .reinforcement_learning.hpo import (
    random_search, bayesian_optimization, 
    grid_search, population_based_bandits_algorithm,
    bayesian_generational_population_based_training,
    differential_evolutionary_hyperband_optimization,
    hyperband_optimization, distributed_evolutionary_algorithm, 
)


def hyperparameter_optimization(opts):
    if opts['cpu_cores'] > 1: update_lock_wait_time(opts['cpu_cores'])    

    # Create directories for optimization results
    try:
        os.makedirs(os.path.join(opts['save_dir'], 'optimization_results'), exist_ok=True)
    except Exception:
        raise Exception("directories to save hyper-parameter optimization output files do not exist and could not be created")

    # Save the optimization configuration
    with open(os.path.join(opts['save_dir'], 'optimization_results', 'optimization_config.json'), 'w') as f:
        json.dump({k: v for k, v in opts.items() if k.startswith(HOP_KEYS)}, f, indent=2)
    
    # Run the selected optimization method
    if opts['hop_method'] == 'gs':
        best_params = grid_search(opts)
    elif opts['hop_method'] == 'rs':
        best_params = random_search(opts)
    elif opts['hop_method'] == 'bo':
        best_params = bayesian_optimization(opts)
    elif opts['hop_method'] == 'hbo':
        best_params = hyperband_optimization(opts)
    elif opts['hop_method'] == 'dea':
        best_params = distributed_evolutionary_algorithm(opts)
    elif opts['hop_method'] == 'dehbo':
        best_params = differential_evolutionary_hyperband_optimization(opts)
    elif opts['hop_method'] == 'pbba':
        best_params = population_based_bandits_algorithm(opts)
    elif opts['hop_method'] == 'bgpbt':
        best_params = bayesian_generational_population_based_training(opts)
    else:
        raise ValueError(f"unknown hyper-parameter optimization method '{opts['hop_method']}'")
    
    # Save the best parameter configuration
    print(f"Best parameters: {best_params}")
    with open(os.path.join(opts['save_dir'], 'optimization_results', 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Run final training with best parameters if requested
    if opts.get('train_best', True):
        print("Running final training with best parameters")
        final_opts = opts.copy()
        for key, value in best_params.items():
            if key == 'learning_rate':
                final_opts['lr_model'] = value
            else:
                final_opts[key] = value
        final_opts['run_name'] = f"{opts['run_name']}_final"
        cost_weights = setup_cost_weights(final_opts)
        return train_reinforcement_learning(final_opts, cost_weights)


def train_meta_reinforcement_learning(opts):
    # Run the selected meta-reinforcement learning method
    #if opts['rl_algorithm'] == 'reinforce':
    if opts['mrl_method'] == 'cb':
        train_func = train_reinforce_over_time_cb
    elif opts['mrl_method'] == 'tdl':
        train_func = train_reinforce_over_time_tdl
    elif opts['mrl_method'] == 'morl':
        train_func = train_reinforce_over_time_morl
    elif opts['mrl_method'] == 'rwa':
        train_func = train_reinforce_over_time_rwa
    else:
        print(f"ERROR: unknown meta-reinforcement learning method '{opts['mrl_method']}'")
        return 1
    #else:
    #    raise ValueError(f"Unknown reinforcement learning algorithm: {opts['rl_algorithm']}")
    return train_reinforcement_learning(opts, train_func)

def train_reinforcement_learning(opts, train_function, cost_weights=None):
    if cost_weights is None: cost_weights = setup_cost_weights(opts)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts['no_tensorboard']:
        tb_logger = TbLogger(
            os.path.join(ROOT_DIR, opts['log_dir'], 
                        "{}_{}".format(opts['problem'], opts['graph_size']), 
                        opts['run_name']))

    # Create directory for saving model checkpoints
    try:
        os.makedirs(os.path.join(ROOT_DIR, opts['save_dir']), exist_ok=True)
    except Exception:
        raise Exception("directories to save model checkpoints do not exist and could not be created")

    # Save arguments so exact configuration can always be found
    with open(os.path.join(ROOT_DIR, opts['save_dir'], "args.json"), 'w') as f:
        json.dump(opts, f, indent=True)

    # Set the device
    use_cuda = torch.cuda.is_available() and not opts['no_cuda']
    opts['device'] = torch.device("cpu" if not use_cuda else "cuda:0")

    # Figure out what's the problem
    problem = load_problem(opts['problem'])

    # Load data from load_path
    data_loader = load_data(opts['load_path'], opts['resume'])

    # Initialize the model and the baseline
    model, baseline = setup_model_and_baseline(problem, data_loader, use_cuda, opts)

    # Setup the optimizer and the learning rate scheduler
    optimizer, lr_scheduler = setup_optimizer_and_lr_scheduler(model, baseline, data_loader, opts)

    # Optionally configure weights and biases
    if opts['wandb_mode'] != 'disabled':
        wandb.init(
            project="wsmart_route",
            name=opts['run_name'],
            config={key: value for key, value in opts.items() if value is not None},
            mode=opts['wandb_mode'],
            tags=[opts['model'], opts['encoder']]
        )
        wandb.watch(model)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts['graph_size'], num_samples=opts['val_size'], area=opts['area'], 
        waste_type=opts['waste_type'], dist_matrix_path=opts['dm_filepath'],
        filename=opts['val_dataset'], distribution=opts['data_distribution'], 
        vertex_strat=opts['vertex_method'], dist_strat=opts['distance_method'],
        number_edges=opts['edge_threshold'], edge_strat=opts['edge_method'],
        focus_graph=opts['focus_graph'], focus_size=opts['eval_focus_size']
    )
    if opts['resume']:
        epoch_resume = int(os.path.splitext(
            os.path.split(opts['resume'])[-1])[0].split("-")[1])
        torch.set_rng_state(data_loader['rng_state'])
        if use_cuda:
            torch.cuda.set_rng_state_all(data_loader['cuda_rng_state'])

        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts['epoch_start'] = epoch_resume + 1

    try:
        if opts['eval_only']:
            validate(model, val_dataset, cost_weights, opts)
        else:
            # Set the scaler and train model
            assert opts['enable_scaler'] and use_cuda or not opts['enable_scaler'], \
            "Scaler for automatic mixed precision can only be used with CUDA GPU(s)"
            scaler = torch.cuda.amp.GradScaler() if opts['enable_scaler'] else None
            if opts['train_time']:
                train_function(model, optimizer, baseline, lr_scheduler, scaler, 
                                val_dataset, problem, tb_logger, cost_weights, opts)
            else:
                for epoch in range(opts['epoch_start'], opts['epoch_start'] + opts['n_epochs']):
                    train_function(model, optimizer, baseline, lr_scheduler, scaler, epoch, 
                                    val_dataset, problem, tb_logger, cost_weights, opts)
        if opts['wandb_mode'] != 'disabled': wandb.finish()
    except Exception as e:
        raise Exception(f"failed to train model with exception due to {repr(e)}")
    return model


def run_training(args):
    comm, opts = args
    exit_code = 0
    try:
        # Set the random seed and execute the program
        random.seed(opts['seed'])
        np.random.seed(opts['seed'])
        torch.manual_seed(opts['seed'])
        if comm == 'train':
            #if opts['rl_algorithm'] == 'reinforce':
            train_func = train_reinforce_over_time if opts['train_time'] else train_reinforce_epoch
            #else:
            #    raise ValueError(f"Unknown reinforcement learning algorithm: {opts['rl_algorithm']}")
            train_reinforcement_learning(opts, train_func)
        elif comm == 'mrl_train':
            train_meta_reinforcement_learning(opts)
        elif comm == 'hp_optim':
            hyperparameter_optimization(opts)
    except Exception as e:
        print(e)
        exit_code = 1
        traceback.print_exc(file=sys.stdout)
    finally:
        sys.stdout.flush()
        sys.exit(exit_code)


if __name__ =="__main__":
    run_training(parse_params())