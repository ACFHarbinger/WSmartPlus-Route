"""
Hyperparameter Optimization Module

This module implements advanced hyperparameter optimization (HPO) algorithms for tuning reinforcement
learning agents in the WSmart-Route framework. It provides a unified interface for multiple optimization
strategies, from classical grid search to state-of-the-art evolutionary hyperband methods.

Available Algorithms:
    - Grid Search: Exhaustive search over a specified parameter grid.
    - Random Search: Samples hyperparameters randomly from a continuous search space.
    - Bayesian Optimization (Optuna): Intelligent search using Tree-structured Parzen Estimator (TPE).
    - Hyperband Optimization: Adaptive resource allocation using Successive Halving.
    - Distributed Evolutionary Algorithm (DEAP): Genetic algorithm for non-convex landscapes.
    - Differential Evolution Hyperband (DEHB): State-of-the-art combination of DE and Hyperband.
"""

import json
import os
import random
import time
from functools import partial

import joblib
import numpy as np
import optuna
import ray
import torch
from deap import algorithms, base, creator, tools
from optuna.visualization import (
    plot_intermediate_values,
    plot_optimization_history,
    plot_param_importances,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from ray.tune.search import BasicVariantGenerator

from logic.src.pipeline.reinforcement_learning.core.epoch import (
    validate_update,
)
from logic.src.pipeline.reinforcement_learning.hyperparameter_optimization.dehb import (
    DifferentialEvolutionHyperband,
    get_config_space,
)
from logic.src.pipeline.reinforcement_learning.worker_train import (
    train_reinforce_epoch,
    train_reinforce_over_time,
)
from logic.src.pipeline.simulator.network import compute_distance_matrix
from logic.src.utils.data_utils import load_focus_coords
from logic.src.utils.definitions import ROOT_DIR
from logic.src.utils.functions import load_data, load_problem
from logic.src.utils.setup_utils import (
    setup_cost_weights,
    setup_model_and_baseline,
    setup_optimizer_and_lr_scheduler,
)


def compute_focus_dist_matrix(graph_size, focus_graph, area="Rio Maior", method="og"):
    """
    Computes the distance matrix for a focus graph area.

    Args:
        graph_size (int): Size of the graph.
        focus_graph (str): Path or identifier for the focus graph.
        area (str, optional): Geographical area. Defaults to "Rio Maior".
        method (str, optional): Distance computation method. Defaults to "og".

    Returns:
        torch.Tensor: The computed distance matrix.
    """
    coords = load_focus_coords(graph_size, None, area, focus_graph)
    dist_matrix = compute_distance_matrix(coords, method)
    return torch.from_numpy(dist_matrix)


def optimize_model(opts, cost_weights, metric="loss", dist_matrix=None):
    """
    Trains a model with given hyperparameters and returns evaluation metrics.

    This is the core worker function used by all HPO algorithms to evaluate a specific configuration.

    Args:
        opts (dict): Options dictionary containing all configuration parameters.
        cost_weights (dict): Dictionary of cost weights to test (w_lost, w_waste, etc.).
        metric (str, optional): Metric to optimize ("loss", "overflows", "kg/km", "both"). Defaults to "loss".
        dist_matrix (torch.Tensor, optional): Precomputed distance matrix for evaluation. Defaults to None.

    Returns:
        tuple: (validation_result, mean_unicost, all_costs)
            - validation_result (float): The primary metric value (e.g., avg cost) used for optimization.
            - mean_unicost (float): Mean unit cost.
            - all_costs (dict): Detailed cost breakdown.

    Raises:
        Exception: If save directories cannot be created.
    """
    # Create directory for saving model checkpoints
    try:
        os.makedirs(os.path.join(ROOT_DIR, opts["save_dir"]), exist_ok=True)
    except Exception:
        raise Exception("directories to save optimization output files do not exist and could not be created")

    # Save arguments so exact configuration can always be found
    with open(os.path.join(ROOT_DIR, opts["save_dir"], "args.json"), "w") as f:
        json.dump({k: v for k, v in opts.items() if k != "device"}, f, indent=True)

    # Set the device
    use_cuda = torch.cuda.is_available() and not opts["no_cuda"]
    opts["device"] = torch.device("cpu" if not use_cuda else "cuda:0")

    # Figure out what's the problem
    problem = load_problem(opts["problem"])

    # Load data from load_path
    data_loader = load_data(opts["load_path"], opts["resume"])

    # Initialize the model and the baseline
    model, baseline = setup_model_and_baseline(problem, data_loader, use_cuda, opts)

    # Setup the optimizer and the learning rate scheduler
    optimizer, lr_scheduler = setup_optimizer_and_lr_scheduler(model, baseline, data_loader, opts)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts["graph_size"],
        num_samples=opts["val_size"],
        filename=opts["val_dataset"],
        area=opts["area"],
        wtype=opts["waste_type"],
        dist_strat=opts["distance_method"],
        distribution=opts["data_distribution"],
        vertex_strat=opts["vertex_method"],
        number_edges=opts["edge_threshold"],
        edge_strat=opts["edge_method"],
        focus_graph=opts["focus_graph"],
        focus_size=opts["eval_focus_size"],
        dist_matrix_path=opts["dm_filepath"],
    )
    if opts["resume"]:
        epoch_resume = int(os.path.splitext(os.path.split(opts["resume"])[-1])[0].split("-")[1])
        torch.set_rng_state(data_loader["rng_state"])
        if use_cuda:
            torch.cuda.set_rng_state_all(data_loader["cuda_rng_state"])

        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts["epoch_start"] = epoch_resume + 1

    # Set the scaler and train model
    assert (
        opts["enable_scaler"] and use_cuda or not opts["enable_scaler"]
    ), "Scaler for automatic mixed precision can only be used with CUDA GPU(s)"
    scaler = torch.cuda.amp.GradScaler() if opts["enable_scaler"] else None
    if opts["train_time"]:
        train_reinforce_over_time(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            scaler,
            val_dataset,
            problem,
            None,
            cost_weights,
            opts,
        )
    else:
        for epoch in range(opts["epoch_start"], opts["epoch_start"] + opts["n_epochs"]):
            train_reinforce_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                scaler,
                epoch,
                val_dataset,
                problem,
                None,
                cost_weights,
                opts,
            )
    return validate_update(model, val_dataset, opts, metric=metric, dist_matrix=dist_matrix)


def distributed_evolutionary_algorithm(opts):
    """
    Genetic algorithm for hyperparameter optimization using DEAP library.

    Use Case: Non-convex, multi-modal optimization landscapes.

    Features:
        - Two-point crossover
        - Polynomial bounded mutation
        - Tournament selection
        - Configurable population size and generations

    Args:
        opts (dict): Options dictionary. Must contain:
            - 'hop_range': [min, max] range for hyperparameters.
            - 'n_pop': Population size.
            - 'n_gen': Number of generations.
            - 'eta', 'indpb', 'tournsize', 'cxpb', 'mutpb': GA parameters.

    Returns:
        dict: The best hyperparameter configuration found.
    """

    def __individual_to_opts(individual, opts):
        """Convert a GA individual into a full opts dictionary."""
        w_lost, w_waste, w_length, w_overflows = individual
        new_opts = {key: val for key, val in opts.items() if "w_" not in key}
        if new_opts["problem"] == "wcvrp":
            new_opts["w_lost"] = w_lost
            new_opts["w_waste"] = w_waste
            new_opts["w_length"] = w_length
            new_opts["w_overflows"] = w_overflows
        return new_opts

    def _create_hyperparameter_configuration(opts):
        """Sample a random hyperparameter configuration for GA initialization."""
        if opts["problem"] == "wcvrp":
            wl = random.uniform(opts["hop_range"][0], opts["hop_range"][1])
            p = random.uniform(opts["hop_range"][0], opts["hop_range"][1])
            loss = random.uniform(opts["hop_range"][0], opts["hop_range"][1])
            o = random.uniform(opts["hop_range"][0], opts["hop_range"][1])
            return [wl, p, loss, o]

    def _fitness_function(individual, cost_weights, opts):
        """Evaluate an individual's fitness using the training loop."""
        new_opts = __individual_to_opts(individual, opts)
        avg_cost, _, _ = optimize_model(new_opts, cost_weights)
        return (avg_cost,)

    # Define fitness function (to minimize validation loss)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Create the DEAP toolbox and define the hyperparameters to optimizer
    toolbox = base.Toolbox()
    # toolbox.register("attr_float", random.uniform, opts['hop_range'][0], opts['hop_range'][1])
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        partial(_create_hyperparameter_configuration, opts),
    )

    # Create population and define genetic operators
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=opts["hop_range"][0],
        up=opts["hop_range"][1],
        eta=opts["eta"],
        indpb=opts["indpb"],
    )
    toolbox.register("select", tools.selTournament, tournsize=opts["tournsize"])

    # Register the fitness function (evaluation metric)
    toolbox.register(
        "evaluate",
        partial(_fitness_function, cost_weights=setup_cost_weights(opts), opts=opts),
    )

    # Initialize hyperparameter optimization
    hof = tools.HallOfFame(1)
    population = toolbox.population(n=opts["n_pop"])

    # Run the parameter optimization
    start_time = time.time()
    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=opts["cxpb"],
        mutpb=opts["mutpb"],
        ngen=opts["n_gen"],
        stats=None,
        halloffame=hof,
        verbose=opts["verbose"] > 0,
    )

    print("Distributed evolutionary algorithm completed in {:.2f} seconds".format(time.time() - start_time))
    best_individual = tools.selBest(population, 1)[0]
    fitness = best_individual.fitness.values[0]
    best_params = __individual_to_opts(best_individual, opts)
    print(f"Best individual: {best_individual}, Fitness: {fitness}")
    return best_params


def bayesian_optimization(opts):
    """
    Uses Tree-structured Parzen Estimator (TPE) for intelligent hyperparameter search with pruning via Optuna.

    Use Case: Expensive objective functions where you need sample-efficient optimization.

    Features:
        - TPE sampler for smart exploration.
        - Median pruning for early termination of poor trials.
        - Visualization (optimization history, parameter importance, intermediate values).
        - SQLite-based persistence for resumable experiments.

    Args:
        opts (dict): Options dictionary.

    Returns:
        dict: The best hyperparameter configuration found.
    """
    # Create a directory to save optimization results
    opt_dir = os.path.join(ROOT_DIR, opts["save_dir"], "optuna_opt")
    try:
        os.makedirs(opt_dir, exist_ok=True)
    except Exception:
        raise Exception("directories to save bayesian optimization output files do not exist and could not be created")

    # Create or load a study
    study_name = f"{opts['problem']}_{opts['graph_size']}_optimization"
    storage_name = f"sqlite:///{os.path.join(opt_dir, 'optuna_study.db')}"
    try:

        def _objective(trial):
            """Optuna objective that trains and returns the validation metric."""
            # Define the hyperparameters for Optuna to optimize
            if opts["problem"] == "wcvrp":
                cost_weights = {
                    "w_lost": trial.suggest_float("w_lost", opts["hop_range"][0], opts["hop_range"][1]),
                    "w_waste": trial.suggest_float("w_waste", opts["hop_range"][0], opts["hop_range"][1]),
                    "w_length": trial.suggest_float("w_length", opts["hop_range"][0], opts["hop_range"][1]),
                    "w_overflows": trial.suggest_float("w_overflows", opts["hop_range"][0], opts["hop_range"][1]),
                }

            print(f"Trial {trial.number}: Evaluating with cost weights: {cost_weights}")

            # Create a trial-specific run name
            trial_opts = opts.copy()
            trial_opts["run_name"] = f"{opts['run_name']}_trial_{trial.number}"

            # Run the training with these weights and get validation performance
            val_result, _, _ = optimize_model(trial_opts, cost_weights=cost_weights)

            # Report intermediate values if your train_model supports callbacks
            # trial.report(intermediate_value, epoch)

            # Return the validation metric (lower is better)
            return val_result

        # Create a pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=opts["n_startup_trials"],
            n_warmup_steps=opts["n_warmup_steps"],
            interval_steps=opts["interval_steps"],
        )

        # Create or load a study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=opts["seed"]),
            pruner=pruner,
        )

        # Run the optimization
        start_time = time.time()
        study.optimize(
            _objective,
            n_trials=opts.get("n_trials", 20),
            timeout=opts.get("timeout", None),
            show_progress_bar=True,
        )

        print("Bayesian optimization study statistics: ")
        print("- Runtime: {:.2f} seconds".format(time.time() - start_time))
        print(f"- Number of finished trials: {len(study.trials)}")
        pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"- Number of pruned trials: {pruned_count}")
        complete_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"- Number of complete trials: {complete_count}")

        print("- Best trial:")
        trial = study.best_trial
        print(f" - Value: {trial.value}")
        print(" - Params: ")
        for key, value in trial.params.items():
            print(f"   - {key}: {value}")

        # Save study for later analysis
        joblib.dump(study, os.path.join(opt_dir, "study.pkl"))

        # Visualization
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(opt_dir, "optimization_history.png"))

        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(opt_dir, "param_importances.png"))

        # Plot intermediate values (shows pruning visually)
        fig3 = plot_intermediate_values(study)
        fig3.write_image(os.path.join(opt_dir, "intermediate_values.png"))

        # Save the optimization results in JSON format
        with open(os.path.join(opt_dir, "optimization_output.json"), "w") as f:
            opt_result = {
                "best_params": trial.params,
                "best_value": trial.value,
                "n_trials": len(study.trials),
                "n_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "trial_values": [t.value for t in study.trials if t.value is not None],
                "datetime_start": study.datetime_start.isoformat(),
                "datetime_complete": study.datetime_complete.isoformat() if study.datetime_complete else None,
            }
            json.dump(opt_result, f, indent=2)

        return trial.params
    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        raise


def _ray_tune_trainable(opts, config, checkpoint_dir=None):
    """
    Internal wrapper function for Ray Tune trainable API.
    """
    # Update opts with the hyperparameters from config
    current_opts = opts.copy()
    if current_opts["problem"] == "wcvrp":
        current_opts["w_lost"] = config["w_lost"]
        current_opts["w_waste"] = config["w_waste"]
        current_opts["w_length"] = config["w_length"]
        current_opts["w_overflows"] = config["w_overflows"]

    # Create unique run name and dirs for this trial
    trial_id = tune.get_trial_id()
    current_opts["run_name"] = f"{current_opts['run_name']}_{trial_id}"
    current_opts["log_dir"] = os.path.join(current_opts["log_dir"], trial_id)
    current_opts["save_dir"] = os.path.join(current_opts["save_dir"], trial_id)

    # Set epochs to a smaller number for early iterations of Hyperband
    current_opts["n_epochs"] = current_opts.get("hop_epochs", 10)

    # Checkpoint loading if resuming
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        current_opts["resume"] = checkpoint_path

    validation_result, _, _ = optimize_model(current_opts, config)

    # Assuming validate() returns a dict with metrics or a scalar value
    if isinstance(validation_result, dict):
        tune.report(**validation_result)
    else:
        tune.report(score=validation_result)
    return


def hyperband_optimization(opts):
    """
    Adaptive resource allocation using the Hyperband algorithm with Ray Tune.

    Use Case: When you want to efficiently allocate computational budget across many configurations.

    Features:
        - Successive halving with adaptive fidelity.
        - Ray Tune integration for distributed execution.
        - Configurable reduction factor (eta).

    Args:
        opts (dict): Options dictionary.

    Returns:
        dict: The best hyperparameter configuration found.
    """
    # Initialize Ray
    gpu_num = 0 if not torch.cuda.is_available() or opts["no_cuda"] else torch.cuda.device_count()
    ray.init(
        num_cpus=opts.get("cpu_cores", None),
        num_gpus=gpu_num,
        local_mode=opts.get("local_mode", False),
    )

    # Define the hyperparameter search space
    config = {
        "w_lost": tune.uniform(opts["hop_range"][0], opts["hop_range"][1]),
        "w_waste": tune.uniform(opts["hop_range"][0], opts["hop_range"][1]),
        "w_length": tune.uniform(opts["hop_range"][0], opts["hop_range"][1]),
        "w_overflows": tune.uniform(opts["hop_range"][0], opts["hop_range"][1]),
        # "optimizer": tune.choice(["adam", "sgd"]),
        # "weight_decay": tune.loguniform(1e-5, 1e-2),
        # "dropout": tune.uniform(0.0, 0.5)
    }

    # Configure the Hyperband scheduler
    metric = opts.get("metric", "loss")
    mode = "max" if metric in ["mean_reward", "episode_reward_mean"] else "min"
    hyperband = HyperBandScheduler(
        time_attr="training_iteration",
        metric=metric,
        mode=mode,
        max_t=opts.get("max_tres", opts["hop_epochs"]),
        reduction_factor=opts.get("reduction_factor", 3),
    )

    # Run the hyperparameter optimization
    start_time = time.time()
    analysis = tune.run(
        _ray_tune_trainable,
        opts=opts,
        config=config,
        scheduler=hyperband,
        num_samples=opts.get("num_samples", 20),
        resources_per_trial={"cpu": opts.get("cpu_cores", 1), "gpu": gpu_num},
        local_dir=os.path.join(opts["log_dir"], "hyperband_output"),
        checkpoint_at_end=True,
        verbose=opts.get("verbose", 2),
        search_alg=BasicVariantGenerator(random_state=opts["seed"]),
    )

    # Get the best configuration
    best_config = analysis.get_best_config(metric=metric, mode=mode)
    best_trial = analysis.get_best_trial(metric=metric, mode=mode)
    print("Hyperband completed in {:.2f} seconds".format(time.time() - start_time))
    print("Best trial metrics: ", best_trial.last_result)

    # Optionally shut down Ray
    ray.shutdown()
    return best_config


def random_search(opts):
    """
    Samples hyperparameters randomly from a continuous search space using Ray Tune.

    Use Case: Large search spaces where grid search is infeasible; good baseline for comparison.

    Features:
        - Uniform sampling from continuous ranges.
        - Configurable number of trials.
        - Supports parallel execution.

    Args:
        opts (dict): Options dictionary.

    Returns:
        dict: The best hyperparameter configuration found.
    """
    # Initialize Ray
    gpu_num = 0 if not torch.cuda.is_available() or opts["no_cuda"] else torch.cuda.device_count()
    ray.init(
        num_cpus=opts.get("cpu_cores", None),
        num_gpus=gpu_num,
        local_mode=opts.get("local_mode", False),
    )

    # Define the hyperparameter search space
    if opts["problem"] == "wcvrp":
        config = {
            "w_lost": tune.uniform(opts["hop_range"][0], opts["hop_range"][1]),
            "w_waste": tune.uniform(opts["hop_range"][0], opts["hop_range"][1]),
            "w_length": tune.uniform(opts["hop_range"][0], opts["hop_range"][1]),
            "w_overflows": tune.uniform(opts["hop_range"][0], opts["hop_range"][1]),
        }

    gpu_num = 0 if not torch.cuda.is_available() or opts["no_cuda"] else torch.cuda.device_count()

    # Run the random hyperparameter search
    start_time = time.time()
    analysis = tune.run(
        _ray_tune_trainable,
        opts=opts,
        config=config,
        num_samples=opts.get("num_samples", 20),
        resources_per_trial={"cpu": opts.get("cpu_cores", 1), "gpu": gpu_num},
        local_dir=os.path.join(opts["log_dir"], "random_search_output"),
        checkpoint_at_end=True,
        search_alg=BasicVariantGenerator(random_state=opts["seed"]),
        verbose=opts.get("verbose", 2),
        max_failures=opts.get("max_failures", 3),
    )

    # Get the best configuration and trial
    metric = opts.get("metric", "loss")
    mode = "max" if metric in ["mean_reward", "episode_reward_mean"] else "min"
    best_config = analysis.get_best_config(metric=metric, mode=mode)
    best_trial = analysis.get_best_trial(metric=metric, mode=mode)
    print("Random search completed in {:.2f} seconds".format(time.time() - start_time))
    print("Best trial metrics: ", best_trial.last_result)

    # Optionally shut down Ray
    ray.shutdown()
    return best_config


def grid_search(opts):
    """
    Exhaustive search over a specified parameter grid using Ray Tune with ASHA scheduling.

    Use Case: Small hyperparameter spaces where you want to explore all combinations systematically.

    Features:
        - Parallel execution via Ray.
        - Early stopping with ASHA scheduler.
        - Progress reporting with Ray CLIReporter.

    Args:
        opts (dict): Options dictionary.

    Returns:
        dict: The best hyperparameter configuration found.
    """

    def tune_hyperparameters(config, opts):
        """Train and report a grid-search configuration to Ray Tune."""
        # Update opts with tuning parameters
        tune_opts = opts.copy()
        for key, value in config.items():
            tune_opts[key] = value

        # Set a unique run name based on trial ID
        trial_id = ray.train.get_context().get_trial_id()
        tune_opts["run_name"] = f"{opts['run_name']}_tune_{trial_id}"

        # Make sure the save paths are unique per trial
        tune_opts["save_dir"] = os.path.join(opts["save_dir"], f"trial_{trial_id}")
        if not opts["no_tensorboard"]:
            tune_opts["log_dir"] = os.path.join(opts["log_dir"], f"trial_{trial_id}")

        # Run training
        result, _, _ = optimize_model(tune_opts, cost_weights=config)

        # Report the result to Ray Tune
        # Assuming lower validation metrics are better
        tune.report(validation_metric=result)
        return

    # Setup parameter grid
    default_wcost_grid = [0.0, 0.5, 1]
    if opts["problem"] == "wcvrp":
        param_grid = {
            "w_lost": opts.get("grid", default_wcost_grid),
            "w_waste": opts.get("grid", default_wcost_grid),
            "w_length": opts.get("grid", default_wcost_grid),
            "w_overflows": opts.get("grid", default_wcost_grid),
        }

    # Initialize Ray (with object store memory limit if needed)
    ray.init(ignore_reinit_error=True)

    # Set up the parameter search space for Ray Tune
    # Convert param_grid to Ray Tune's grid_search format
    tune_config = {k: tune.grid_search(v) for k, v in param_grid.items()}

    # Setup search algorithm and scheduler
    # For grid search, we don't need a special search algorithm
    # ASHA scheduler can help terminate poorly performing trials early
    max_t_epochs = opts.get("max_tres", opts["hop_epochs"])
    scheduler = ASHAScheduler(
        metric="validation_metric",
        mode="min",  # Assuming lower is better
        max_t=max_t_epochs,
        grace_period=min(5, max_t_epochs // 3),
        reduction_factor=2,
    )

    # Setup progress reporter
    reporter = tune.CLIReporter(
        metric_columns=["validation_metric", "training_iteration"],
        parameter_columns=list(param_grid.keys()),
    )

    # Create a trainable by partially applying the opts to train_with_tune
    trainable = partial(tune_hyperparameters, opts=opts)
    gpu_num = 0 if not torch.cuda.is_available() or opts["no_cuda"] else torch.cuda.device_count()
    print(f"Starting Ray Tune grid search with parameter space: {param_grid}")
    print(f"This will run {np.prod([len(v) for v in param_grid.values()])} trials")

    # Run the grid hyperparameter search
    start_time = time.time()
    result = tune.run(
        trainable,
        resources_per_trial={"cpu": opts["cpu_cores"], "gpu": gpu_num},
        config=tune_config,
        num_samples=opts["num_samples"],
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=os.path.join(opts["save_dir"], "grid_search_output"),
        name=f"{opts['run_name']}_grid_search",
        keep_checkpoints_num=1,
        checkpoint_score_attr="min-validation_metric",
        max_concurrent_trials=opts["max_conc"],
        fail_fast=True,
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
    """
    State-of-the-art combination of Differential Evolution and Hyperband for efficient multi-fidelity optimization.

    Use Case: When you have access to multi-fidelity evaluations (e.g., training for different numbers of epochs).

    Features:
        - Asynchronous execution with Dask.
        - Multi-fidelity optimization.
        - Adaptive population management.
        - Checkpoint saving for long-running experiments.

    Args:
        opts (dict): Options dictionary.

    Returns:
        dict: The best hyperparameter configuration found.
    """

    def _objective_function(config, fidelity, **kwargs):
        """Evaluate a DEHB configuration and return fitness/cost info."""
        metric = kwargs.get("metric", "loss")
        dist_matrix = kwargs.get("dist_matrix", None)
        # Update the options with the current configuration
        for key, value in config.items():
            if key == "learning_rate":
                opts["lr_model"] = value
            else:
                opts[key] = value

        if fidelity is None:
            fidelity = opts["n_epochs"]

        cost_weights = setup_cost_weights(opts)
        avg_costs, avg_ucosts, _ = optimize_model(opts, cost_weights, metric, dist_matrix)
        result = {
            "fitness": avg_costs,
            "cost": avg_ucosts,
            "info": {"fidelity": fidelity},
        }
        return result

    dist_matrix = None
    if opts["metric"] in ["kg/km", "both"]:
        assert opts["focus_graph"] is not None and opts["eval_focus_size"] == opts["val_size"]
        dist_matrix = compute_focus_dist_matrix(opts["graph_size"], opts["focus_graph"], opts["area"])

    # Initialize DEHB
    dehb = DifferentialEvolutionHyperband(
        cs=get_config_space(opts),
        f=_objective_function,
        mutation_factor=opts.get("mutpb", 0.5),
        crossover_prob=opts.get("cxpb", 0.5),
        min_fidelity=opts.get("hop_epochs", 1),
        max_fidelity=opts.get("max_tres", 10),
        eta=opts.get("eta", 3),
        n_workers=opts.get("cpu_cores", 4),
        output_path=os.path.join(opts["save_dir"], "dehb_output"),
        log_level="INFO",
    )
    dehb._init_population(opts.get("n_pop", 20))

    # Run DEHB optimization
    traj, runtime, history = dehb.run(fevals=opts.get("fevals", 100), metric=opts["metric"], dist_matrix=dist_matrix)
    print("Differential evolution hyperband completed in {:.2f} seconds".format(runtime))

    # Get the best configuration and fitness
    best_config, best_fitness = dehb.get_incumbents()
    print("Best Configuration:", best_config)
    print("Best Fitness:", best_fitness)
    return best_config
