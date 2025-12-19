import os
import ray
import sys
import json
import math
import time
import torch
import random
import optuna
import joblib
import logging
import traceback
import numpy as np
import pandas as pd

from ray import tune
from tqdm import tqdm
from functools import partial
from datetime import datetime
from deap import base, creator, tools, algorithms
from ray.tune.search import BasicVariantGenerator
from ray.tune.schedulers import (
    PopulationBasedTraining,
    AsyncHyperBandScheduler, 
    HyperBandScheduler, ASHAScheduler, 
)
from optuna.visualization import (
    plot_param_importances, 
    plot_intermediate_values,
    plot_optimization_history, 
)

from logic.src.utils.definitions import ROOT_DIR
from logic.src.utils.data_utils import load_focus_coords
from logic.src.utils.functions import load_data, load_problem, move_to
from logic.src.utils.setup_utils import (
    setup_cost_weights, 
    setup_model_and_baseline, 
    setup_optimizer_and_lr_scheduler
)
from .epoch import set_decode_type, get_inner_model
from ..simulator.network import compute_distance_matrix
from .reinforce import (
    train_reinforce_epoch, train_reinforce_over_time
)
from .hyperparameter_optimization import (
    DifferentialEvolutionHyperband, get_config_space,
    #PB2, BGPBT, BGPBTArch, Brax
)


def compute_focus_dist_matrix(graph_size, focus_graph, area="Rio Maior", method="og"):
    coords = load_focus_coords(graph_size, None, area, focus_graph)
    dist_matrix = compute_distance_matrix(coords, method)
    return torch.from_numpy(dist_matrix)


def optimize_model(opts, cost_weights, metric="loss", dist_matrix=None):
    # Create directory for saving model checkpoints
    try:
        os.makedirs(os.path.join(ROOT_DIR, opts['save_dir']), exist_ok=True)
    except Exception:
        raise Exception("directories to save optimization output files do not exist and could not be created")

    # Save arguments so exact configuration can always be found
    with open(os.path.join(ROOT_DIR, opts['save_dir'], "args.json"), 'w') as f:
        json.dump({k: v for k, v in opts.items() if k != 'device'}, f, indent=True)

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

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts['graph_size'], num_samples=opts['val_size'], filename=opts['val_dataset'],
        area=opts['area'], wtype=opts['waste_type'], dist_strat=opts['distance_method'],
        distribution=opts['data_distribution'], vertex_strat=opts['vertex_method'],
        number_edges=opts['edge_threshold'], edge_strat=opts['edge_method'],
        focus_graph=opts['focus_graph'], focus_size=opts['eval_focus_size'],
        dist_matrix_path=opts['dm_filepath']
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

    # Set the scaler and train model
    assert opts['enable_scaler'] and use_cuda or not opts['enable_scaler'], \
    "Scaler for automatic mixed precision can only be used with CUDA GPU(s)"
    scaler = torch.cuda.amp.GradScaler() if opts['enable_scaler'] else None
    if opts['train_time']:
        train_reinforce_over_time(model, optimizer, baseline, lr_scheduler, scaler, 
                                val_dataset, problem, None, cost_weights, opts)
    else:
        for epoch in range(opts['epoch_start'], 
                            opts['epoch_start'] + opts['n_epochs']):
            train_reinforce_epoch(model, optimizer, baseline, lr_scheduler, scaler, 
                                epoch, val_dataset, problem, None, cost_weights, opts)
    return validate(model, val_dataset, metric, dist_matrix, opts)


def validate(model, dataset, metric, dist_matrix, opts):
    def eval_model_bat(bat, dist_matrix):
        with torch.no_grad():
            ucost, c_dict, attn_dict = get_inner_model(model).compute_batch_sim(move_to(bat, opts['device']), move_to(dist_matrix, opts['device']))
        return ucost, c_dict, attn_dict

    # Put in greedy evaluation mode!
    print('Validating...')
    set_decode_type(model, "greedy")
    all_costs = {'overflows': [], 'kg': [], 'km': []}
    all_ucosts = move_to(torch.tensor([]), opts['device'])
    attention_dict = {'attention_weights': [], 'graph_masks': []}
    model.eval()
    for bat in tqdm(torch.utils.data.DataLoader(dataset, batch_size=opts['eval_batch_size'], pin_memory=True), disable=opts['no_progress_bar']):
        ucost, cost_dict, attn_dict = eval_model_bat(bat, dist_matrix)
        for key in attention_dict.keys():
            attention_dict[key].append(attn_dict[key])

        all_ucosts = torch.cat((all_ucosts, ucost), 0)
        for key, val in cost_dict.items():
            all_costs[key].append(val)

    for key, val in attention_dict.items():
        attention_dict[key] = torch.cat(val)

    for key, val in all_costs.items():
        all_costs[key] = torch.cat(val)

    if metric == 'overflows':
        cost = all_costs[metric]
    elif metric == 'kg/km':
        cost = all_costs['kg'] / all_costs['km']
    else:
        assert metric == 'both'
        cost = all_costs['kg'] / all_costs['km'] - all_costs['overflows']
    
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    return avg_cost, all_ucosts.mean(), all_costs


def distributed_evolutionary_algorithm(opts):
    def __individual_to_opts(individual, opts):
        w_lost, w_waste, w_length, w_overflows = individual
        new_opts = {key: val for key, val in opts.items() if 'w_' not in key}
        if new_opts['problem'] == 'wcvrp':
            new_opts['w_lost'] = w_lost
            new_opts['w_waste'] = w_waste
            new_opts['w_length'] = w_length
            new_opts['w_overflows'] = w_overflows
        return new_opts

    def _create_hyperparameter_configuration(opts):
        if opts['problem'] == 'wcvrp':
            wl = random.uniform(opts['hop_range'][0], opts['hop_range'][1]) 
            p = random.uniform(opts['hop_range'][0], opts['hop_range'][1])
            l = random.uniform(opts['hop_range'][0], opts['hop_range'][1])
            o = random.uniform(opts['hop_range'][0], opts['hop_range'][1])
            return [wl, p, l, o]

    def _fitness_function(individual, cost_weights, opts):
        new_opts = __individual_to_opts(individual, opts)
        avg_cost, _, _ = optimize_model(new_opts, cost_weights)
        return avg_cost,

    # Define fitness function (to minimize validation loss)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Create the DEAP toolbox and define the hyperparameters to optimizer
    toolbox = base.Toolbox()
    #toolbox.register("attr_float", random.uniform, opts['hop_range'][0], opts['hop_range'][1])
    toolbox.register("individual", tools.initIterate, creator.Individual, 
        partial(_create_hyperparameter_configuration, opts))
    
    # Create population and define genetic operators
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=opts['hop_range'][0], 
                    up=opts['hop_range'][1], eta=opts['eta'], indpb=opts['indpb'])
    toolbox.register("select", tools.selTournament, tournsize=opts['tournsize'])

    # Register the fitness function (evaluation metric)
    toolbox.register("evaluate", 
        partial(_fitness_function, cost_weights=setup_cost_weights(opts), opts=opts))
    
    # Initialize hyperparameter optimization
    hof = tools.HallOfFame(1)
    population = toolbox.population(n=opts['n_pop'])
    
    # Run the parameter optimization
    start_time = time.time()
    algorithms.eaSimple(population, toolbox, cxpb=opts['cxpb'], mutpb=opts['mutpb'], 
                        ngen=opts['n_gen'], stats=None, halloffame=hof, verbose=opts['verbose'] > 0)
    
    print("Distributed evolutionary algorithm completed in {:.2f} seconds".format(time.time() - start_time))
    best_individual = tools.selBest(population, 1)[0]
    fitness = best_individual.fitness.values[0]
    best_params = __individual_to_opts(best_individual, opts)
    print(f"Best individual: {best_individual}, Fitness: {fitness}")
    return best_params


def bayesian_optimization(opts):
    # Create a directory to save optimization results
    opt_dir = os.path.join(ROOT_DIR, opts['save_dir'], 'optuna_opt')
    try:
        os.makedirs(opt_dir, exist_ok=True)
    except Exception:
        raise Exception("directories to save bayesian optimization output files do not exist and could not be created")

    # Create or load a study
    study_name = f"{opts['problem']}_{opts['graph_size']}_optimization"
    storage_name = f"sqlite:///{os.path.join(opt_dir, 'optuna_study.db')}"
    try:
        def _objective(trial):
            # Define the hyperparameters for Optuna to optimize
            if opts['problem'] == 'wcvrp':
                cost_weights = {
                    'w_lost': trial.suggest_float('w_lost', opts['hop_range'][0], opts['hop_range'][1]),
                    'w_waste': trial.suggest_float('w_waste', opts['hop_range'][0], opts['hop_range'][1]),
                    'w_length': trial.suggest_float('w_length', opts['hop_range'][0], opts['hop_range'][1]),
                    'w_overflows': trial.suggest_float('w_overflows', opts['hop_range'][0], opts['hop_range'][1]),
                }
            
            print(f"Trial {trial.number}: Evaluating with cost weights: {cost_weights}")
            
            # Create a trial-specific run name
            trial_opts = opts.copy()
            trial_opts['run_name'] = f"{opts['run_name']}_trial_{trial.number}"
            
            # Run the training with these weights and get validation performance
            val_result, _, _ = optimize_model(trial_opts, cost_weights=cost_weights)
            
            # Report intermediate values if your train_model supports callbacks
            # trial.report(intermediate_value, epoch)
            
            # Return the validation metric (lower is better)
            return val_result
        
        # Create a pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=opts['n_startup_trials'],
            n_warmup_steps=opts['n_warmup_steps'],
            interval_steps=opts['interval_steps']
        )

        # Create or load a study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=opts['seed']),
            pruner=pruner
        )
        
        # Run the optimization
        start_time = time.time()
        study.optimize(
            _objective,
            n_trials=opts.get('n_trials', 20),
            timeout=opts.get('timeout', None),
            show_progress_bar=True
        )
        
        print("Bayesian optimization study statistics: ")
        print("- Runtime: {:.2f} seconds".format(time.time() - start_time))
        print(f"- Number of finished trials: {len(study.trials)}")
        print(f"- Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"- Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        
        print(f"- Best trial:")
        trial = study.best_trial
        print(f" - Value: {trial.value}")
        print(f" - Params: ")
        for key, value in trial.params.items():
            print(f"   - {key}: {value}")
        
        # Save study for later analysis
        joblib.dump(study, os.path.join(opt_dir, "study.pkl"))
        
        # Visualization
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(opt_dir, 'optimization_history.png'))
        
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(opt_dir, 'param_importances.png'))
        
        # Plot intermediate values (shows pruning visually)
        fig3 = plot_intermediate_values(study)
        fig3.write_image(os.path.join(opt_dir, 'intermediate_values.png'))
        
        # Save the optimization results in JSON format
        with open(os.path.join(opt_dir, 'optimization_output.json'), 'w') as f:
            opt_result = {
                'best_params': trial.params,
                'best_value': trial.value,
                'n_trials': len(study.trials),
                'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'trial_values': [t.value for t in study.trials if t.value is not None],
                'datetime_start': study.datetime_start.isoformat(),
                'datetime_complete': study.datetime_complete.isoformat() if study.datetime_complete else None
            }
            json.dump(opt_result, f, indent=2)
        
        return trial.params
    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        raise


def _ray_tune_trainable(opts, config, checkpoint_dir=None):
        # Update opts with the hyperparameters from config
        current_opts = opts.copy()
        if current_opts['problem'] == 'wcvrp':
            current_opts['w_lost'] = config['w_lost']
            current_opts['w_waste'] = config['w_waste']
            current_opts['w_length'] = config['w_length']
            current_opts['w_overflows'] = config['w_overflows']
        
        # Create unique run name and dirs for this trial
        trial_id = tune.get_trial_id()
        current_opts['run_name'] = f"{current_opts['run_name']}_{trial_id}"
        current_opts['log_dir'] = os.path.join(current_opts['log_dir'], trial_id)
        current_opts['save_dir'] = os.path.join(current_opts['save_dir'], trial_id)
        
        # Set epochs to a smaller number for early iterations of Hyperband
        current_opts['n_epochs'] = current_opts.get('hop_epochs', 10)
        
        # Checkpoint loading if resuming
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
            current_opts['resume'] = checkpoint_path
        
        validation_result, _, _ = optimize_model(current_opts, config)
        
        # Assuming validate() returns a dict with metrics or a scalar value
        if isinstance(validation_result, dict):
            tune.report(**validation_result)
        else:
            tune.report(score=validation_result)
        return


def hyperband_optimization(opts):
    # Initialize Ray
    gpu_num = 0 if not torch.cuda.is_available() or opts['no_cuda'] \
        else torch.cuda.device_count()
    ray.init(num_cpus=opts.get('cpu_cores', None), 
             num_gpus=gpu_num,
             local_mode=opts.get('local_mode', False))
    
    # Define the hyperparameter search space
    config = {
        "w_lost": tune.uniform(opts['hop_range'][0], opts['hop_range'][1]),
        "w_waste": tune.uniform(opts['hop_range'][0], opts['hop_range'][1]),
        "w_length": tune.uniform(opts['hop_range'][0], opts['hop_range'][1]),
        "w_overflows": tune.uniform(opts['hop_range'][0], opts['hop_range'][1]),
        #"optimizer": tune.choice(["adam", "sgd"]),
        #"weight_decay": tune.loguniform(1e-5, 1e-2),
        #"dropout": tune.uniform(0.0, 0.5)
    }
    
    # Configure the Hyperband scheduler
    metric = opts.get('metric', "loss")
    mode = "max" if metric in ["mean_reward", "episode_reward_mean"] else "min"
    hyperband = HyperBandScheduler(
        time_attr="training_iteration",
        metric=metric,
        mode=mode,
        max_t=opts.get('max_tres', opts['hop_epochs']),
        reduction_factor=opts.get('reduction_factor', 3)
    )
    
    # Run the hyperparameter optimization
    start_time = time.time()
    analysis = tune.run(
        _ray_tune_trainable,
        opts=opts,
        config=config,
        scheduler=hyperband,
        num_samples=opts.get('num_samples', 20),
        resources_per_trial={
            "cpu": opts.get('cpu_cores', 1),
            "gpu": gpu_num
        },
        local_dir=os.path.join(opts['log_dir'], "hyperband_output"),
        checkpoint_at_end=True,
        verbose=opts.get('verbose', 2),
        search_alg=BasicVariantGenerator(random_state=opts['seed'])
    )
    
    # Get the best configuration
    best_config = analysis.get_best_config(
        metric=metric,
        mode=mode
    )
    best_trial = analysis.get_best_trial(
        metric=metric,
        mode=mode
    )
    print("Hyperband completed in {:.2f} seconds".format(time.time() - start_time))
    print("Best trial metrics: ", best_trial.last_result)

    # Optionally shut down Ray
    ray.shutdown()
    return best_config


def random_search(opts):
    # Initialize Ray
    gpu_num = 0 if not torch.cuda.is_available() or opts['no_cuda'] \
        else torch.cuda.device_count()
    ray.init(num_cpus=opts.get('cpu_cores', None), 
            num_gpus=gpu_num, local_mode=opts.get('local_mode', False))
    
    # Define the hyperparameter search space
    if opts['problem'] == 'wcvrp':
        config = {
            "w_lost": tune.uniform(opts['hop_range'][0], opts['hop_range'][1]),
            "w_waste": tune.uniform(opts['hop_range'][0], opts['hop_range'][1]),
            "w_length": tune.uniform(opts['hop_range'][0], opts['hop_range'][1]),
            "w_overflows": tune.uniform(opts['hop_range'][0], opts['hop_range'][1])
        }

    gpu_num = 0 if not torch.cuda.is_available() or opts['no_cuda'] \
        else torch.cuda.device_count()
    
    # Run the random hyperparameter search
    start_time = time.time()
    analysis = tune.run(
        _ray_tune_trainable,
        opts=opts,
        config=config,
        num_samples=opts.get('num_samples', 20),
        resources_per_trial={
            "cpu": opts.get('cpu_cores', 1),
            "gpu": gpu_num
        },
        local_dir=os.path.join(opts['log_dir'], "random_search_output"),
        checkpoint_at_end=True,
        search_alg=BasicVariantGenerator(random_state=opts['seed']),
        verbose=opts.get('verbose', 2),
        max_failures=opts.get('max_failures', 3)
    )
    
    # Get the best configuration and trial
    metric = opts.get('metric', "loss")
    mode = "max" if metric in ["mean_reward", "episode_reward_mean"] else "min"
    best_config = analysis.get_best_config(
        metric=metric,
        mode=mode
    )
    best_trial = analysis.get_best_trial(
        metric=metric,
        mode=mode
    )
    print("Random search completed in {:.2f} seconds".format(time.time() - start_time))
    print("Best trial metrics: ", best_trial.last_result)

    # Optionally shut down Ray
    ray.shutdown()
    return best_config


def grid_search(opts):
    def tune_hyperparameters(config, opts):
        # Update opts with tuning parameters
        tune_opts = opts.copy()
        for key, value in config.items():
            tune_opts[key] = value
        
        # Set a unique run name based on trial ID
        trial_id = ray.train.get_context().get_trial_id()
        tune_opts['run_name'] = f"{opts['run_name']}_tune_{trial_id}"
        
        # Make sure the save paths are unique per trial
        tune_opts['save_dir'] = os.path.join(opts['save_dir'], f"trial_{trial_id}")
        if not opts['no_tensorboard']:
            tune_opts['log_dir'] = os.path.join(opts['log_dir'], f"trial_{trial_id}")
        
        # Run training
        result, _, _ = optimize_model(tune_opts, cost_weights=config)
        
        # Report the result to Ray Tune
        # Assuming lower validation metrics are better
        tune.report(validation_metric=result)
        return

    # Setup parameter grid
    default_wcost_grid = [0.0, 0.5, 1]
    if opts['problem'] == 'wcvrp':
        param_grid = {
            'w_lost': opts.get('grid', default_wcost_grid),
            'w_waste': opts.get('grid', default_wcost_grid),
            'w_length': opts.get('grid', default_wcost_grid),
            'w_overflows': opts.get('grid', default_wcost_grid)
        }

    # Initialize Ray (with object store memory limit if needed)
    ray.init(ignore_reinit_error=True)
    
    # Set up the parameter search space for Ray Tune
    # Convert param_grid to Ray Tune's grid_search format
    tune_config = {k: tune.grid_search(v) for k, v in param_grid.items()}
    
    # Setup search algorithm and scheduler
    # For grid search, we don't need a special search algorithm
    # ASHA scheduler can help terminate poorly performing trials early
    max_t_epochs = opts.get('max_tres', opts['hop_epochs'])
    scheduler = ASHAScheduler(
        metric="validation_metric",
        mode="min",  # Assuming lower is better
        max_t=max_t_epochs,
        grace_period=min(5, max_t_epochs // 3),
        reduction_factor=2
    )
    
    # Setup progress reporter
    reporter = tune.CLIReporter(
        metric_columns=["validation_metric", "training_iteration"],
        parameter_columns=list(param_grid.keys())
    )
    
    # Create a trainable by partially applying the opts to train_with_tune
    trainable = partial(tune_hyperparameters, opts=opts)
    gpu_num = 0 if not torch.cuda.is_available() or opts['no_cuda'] \
        else torch.cuda.device_count()
    print(f"Starting Ray Tune grid search with parameter space: {param_grid}")
    print(f"This will run {np.prod([len(v) for v in param_grid.values()])} trials")
    
    # Run the grid hyperparameter search
    start_time = time.time()
    result = tune.run(
        trainable,
        resources_per_trial={"cpu": opts['cpu_cores'], "gpu": gpu_num},
        config=tune_config,
        num_samples=opts['num_samples'],
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=os.path.join(opts['save_dir'], "grid_search_output"),
        name=f"{opts['run_name']}_grid_search",
        keep_checkpoints_num=1,
        checkpoint_score_attr="min-validation_metric",
        max_concurrent_trials=opts['max_conc'],
        fail_fast=True
    )
    
    # Get the best trial result
    best_trial = result.get_best_trial("validation_metric", "min", "last")
    best_result = best_trial.last_result["validation_metric"]
    print("Grid search completed in {:.2f} seconds".format(time.time() - start_time))
    print(f"Best trial final validation result: {best_result}")
    
    # Optionally shut down Ray
    ray.shutdown()
    return best_trial.config


def differential_evolutionary_hyperband_optimization(opts):
    def _objective_function(config, fidelity, **kwargs):
        metric = kwargs.get("metric", "loss")
        dist_matrix = kwargs.get("dist_matrix", None)
        # Update the options with the current configuration
        for key, value in config.items():
            if key == 'learning_rate':
                opts['lr_model'] = value
            else:
                opts[key] = value

        if fidelity is None:
            fidelity = opts['n_epochs']

        cost_weights = setup_cost_weights(opts)
        avg_costs, avg_ucosts, _ = optimize_model(opts, cost_weights, metric, dist_matrix)
        result = {
            "fitness": avg_costs,
            "cost": avg_ucosts,
            "info": {
                "fidelity": fidelity
            }
        }
        return result
    
    dist_matrix = None
    if opts['metric'] in ['kg/km', 'both']:
        assert opts['focus_graph'] is not None and \
        opts['eval_focus_size'] == opts['val_size']
        dist_matrix = compute_focus_dist_matrix(opts['graph_size'], opts['focus_graph'], opts['area'])

    # Initialize DEHB
    dehb = DifferentialEvolutionHyperband(
        cs=get_config_space(opts),
        f=_objective_function,
        mutation_factor=opts.get('mutpb', 0.5),
        crossover_prob=opts.get('cxpb', 0.5),
        min_fidelity=opts.get('hop_epochs', 1),
        max_fidelity=opts.get('max_tres', 10),
        eta=opts.get('eta', 3),
        n_workers=opts.get('cpu_cores', 4),
        output_path=os.path.join(opts['save_dir'], 'dehb_output'),
        log_level="INFO"
    )
    dehb._init_population(opts.get('n_pop', 20))
    
    # Run DEHB optimization
    traj, runtime, history = dehb.run(
        fevals=opts.get('fevals', 100),
        metric=opts['metric'],
        dist_matrix=dist_matrix
    ) 
    print("Differential evolution hyperband completed in {:.2f} seconds".format(runtime))

    # Get the best configuration and fitness
    best_config, best_fitness = dehb.get_incumbents()
    print("Best Configuration:", best_config)
    print("Best Fitness:", best_fitness)
    return best_config


def population_based_bandits_algorithm(opts):
    raise NotImplementedError
    def _explore(config):
        for key in config.keys():
            if 'w_' in key and config[key] < 0: config[key] = 0.
        return config

    ray.init()
    opts['dir'] = "{}_{}_Size{}_{}_{}".format(
        opts['algo'], opts['method'], str(opts['num_samples']), opts['env_name'], opts['criteria'])
    try:
        os.makedirs('data/'+opts['dir'], exist_ok=True)
    except Exception:
        raise Exception("directories to save output files do not exist and could not be created")

    if opts['problem'] == 'wcvrp':
        hp_names = ['w_lost', 'w_waste', 'w_length', 'w_overflows']
        #mutations = {
        #    "epsilon":  lambda: random.uniform(0.01, 0.5),
        #    "entropy_coeff": lambda: random.uniform(0.001, 0.1),
        #    "lambda": lambda: random.uniform(0.9, 1.0),
        #    "clip_param": lambda: random.uniform(0.1, 0.5),
        #    "lr": lambda: random.uniform(1e-3, 1e-5),
        #    "train_batch_size": lambda: random.randint(int(opts['batchsize'].split("_")[0]), int(opts['batchsize'].split("_")[1])),
        #}

    mode = "max" if opts.get('metric', 'episode_reward_mean') in ["mean_reward", "episode_reward_mean"] else "min"
    mutations = {hp: lambda: random.uniform(opts['hop_range'][0], opts['hop_range'][1]) for hp in hp_names}
    pbt = PopulationBasedTraining(
        time_attr= opts['criteria'],
        metric=opts.get('metric', 'episode_reward_mean'),
        mode=mode,
        perturbation_interval=opts['freq'],
        resample_probability=opts['perturb'],
        quantile_fraction = opts['perturb'], # copy bottom % with top %
        hyperparam_mutations=mutations,
        custom_explore_fn=_explore)
    
    pb2 = PB2(
        time_attr= opts['criteria'],
        metric=opts.get('metric', 'episode_reward_mean'), 
        mode=mode,
        perturbation_interval=opts['freq'],
        resample_probability=0,
        quantile_fraction=opts['perturb'], # copy bottom % with top %
        hyperparam_mutations=mutations,
        custom_explore_fn=_explore)

    max_t_size = opts.get('max_tres', opts['hop_epochs'])
    asha = AsyncHyperBandScheduler(
        time_attr=opts['criteria'],
        metric=opts.get('metric', 'episode_reward_mean'),
        mode=mode,
        grace_period=opts['freq'],
        max_t=max_t_size)
    
    methods = {'pbt': pbt,
               'pb2': pb2,
               'asha': asha}
    
    hp_config = {hp: tune.sample_from(lambda spec: random.uniform(opts['hop_range'][0], opts['hop_range'][1])) for hp in hp_names}
    timelog = str(datetime.date(datetime.now())) + '_' + str(datetime.time(datetime.now()))
    analysis = tune.run(
        opts['algo'],
        name="{}_{}_{}_seed{}".format(timelog, opts['method'], opts['env_name'], str(opts['seed'])),
        scheduler=methods[opts['method']],
        verbose=1,
        num_samples= opts['num_samples'],
        stop={opts['criteria']: max_t_size},
        config= {
            "env": opts['env_name'],
            "log_level": "INFO",
            "seed": opts['seed'],
            "num_gpus": 0,
            "num_workers": opts['num_workers'],
            "horizon": opts['horizon'],
            "rollout_fragment_length": 50,
            "train_batch_size": 500,
            "num_envs_per_worker": 5,
            **hp_config
        }
    )
    all_dfs = analysis.trial_dataframes
    names = list(all_dfs.keys())
    results = pd.DataFrame()    
    for i in range(opts['num_samples']):
        df = all_dfs[names[i]]
        df = df[['timesteps_total', 'time_total_s', 'episodes_total', 
                opts.get('metric', 'episode_reward_mean'), 'info/learner/default_policy/cur_kl_coeff']]
        df['Agent'] = i
        results = pd.concat([results, df]).reset_index(drop=True)
    
    # Get the best trial and its configuration
    best_trial = analysis.get_best_trial(metric=opts.get('metric', 'episode_reward_mean'), mode="max", scope="all")
    best_config = best_trial.config if best_trial else None
    results.to_csv("data/{}/seed{}.csv".format(opts['dir'], str(opts['seed'])))

    total_runtime = analysis.get_all_configs(with_times=True)[-1][1]
    print("Population-based bandits optimization completed in {:.2f} seconds".format(total_runtime))

    # Optionally shut down Ray
    ray.shutdown()
    return best_config


def bayesian_generational_population_based_training(opts):
    raise NotImplementedError
    # JAX by default pre-allocates 90% of the available VRAM and this will lead to OOM if multiple JAX processes are spawned simultaneously
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.05'

    save_path = "{}/data/brax_env_{}_{}_maxIter_{}_{}_ps_{}_quantileFrac_{}_{}_{}_bgpbt".format(ROOT_DIR, opts['env_name'], opts['algo'],
                opts['max_tres'], opts['t_ready'], opts['n_pop'], opts['quantile_fraction'], opts['search_mode'], opts['arch_policy'])
    if opts['run_name'] is not None:
        save_path = os.path.join(
            save_path, f"{opts['run_name']}", f"seed_{opts['seed']}")
    else:
        save_path = os.path.join(save_path, f"seed_{opts['seed']}")
    
    try:
        os.makedirs(save_path, exist_ok=True)
    except Exception:
        raise Exception("directories to save output files do not exist and could not be created")
    
    logging.basicConfig(
        handlers=[
            logging.StreamHandler()
        ],
        level=logging.INFO if opts['verbose'] > 0 else logging.WARNING,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y%m%d %H%M%S',
    )
    env = Brax(
        env_name=opts['env_name'],
        alg_name=opts['algo'],
        seed=opts['seed'],
        do_hpo=True,
        do_nas=opts['search_mode'] == "both",
        log_dir=save_path,
        max_parallel=opts['max_conc']
    )

    if opts['search_mode'] == 'hpo':  # search for hyperparameter only WITHOUT distillation
        hpo = BGPBT(
            env, log_dir=save_path,
            max_timesteps=opts['max_tres'],
            pop_size=opts['n_pop'],
            n_init=opts['n_init'],
            quantile_fraction=opts['quantile_fraction'],
            seed=opts['seed'],
            t_ready=opts['t_ready'],
            t_ready_end=opts['t_ready_end'],
            verbose=opts['verbose'] > 0,
            existing_policy=opts['existing_policy'],
        )
    elif opts['search_mode'] == 'both':
        hpo = BGPBTArch(
            env, log_dir=save_path,
            max_timesteps=opts['max_tres'],
            n_distillation_timesteps=opts['t_distil'],
            pop_size=opts['n_pop'],
            n_init=opts['n_init'],
            quantile_fraction=opts['quantile_fraction'],
            seed=opts['seed'],
            t_ready=opts['t_ready'],
            t_ready_end=opts['t_ready_end'],
            verbose=opts['verbose'] > 0,
            existing_policy=opts['existing_policy'],
            patience=opts['patience'],
            distill_every=opts['distill_every'],
            max_distillation=opts['max_distillation'],
            init_policy='bo' if opts['arch_policy'] == 'search' else 'random',
        )
    
    stats = hpo.run()
    stats.to_csv(os.path.join(save_path, f"stats_seed_{opts['seed']}.csv"))
    print("Bayesian generational population-based optimization completed in {:.2f} seconds".format(stats['time_this_iter_s'].sum()))

    # Extract the best configuration from stats
    best_idx = stats['mean_reward'].idxmax()
    best_config = stats.loc[best_idx].to_dict()

    # Filter to only include hyperparameter-related keys
    hyperparam_keys = [col for col in stats.columns if col not in ['mean_reward', 'timesteps', 'iteration']]
    best_config = {k: best_config[k] for k in hyperparam_keys if k in best_config}
    return best_config

def merge_with_training_args(training_args, hop_args):
    # Only add HOP args that don't already exist or that are explicitly specified
    merged = training_args.copy()
    for key, value in hop_args.items():
        if key not in merged or value is not None:
            merged[key] = value
    return merged