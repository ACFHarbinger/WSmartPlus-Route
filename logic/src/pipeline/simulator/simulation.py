"""
High-Level Simulation Orchestration and Parallelization.

This module provides the top-level entry points for running simulations:
single runs, sequential batches, and parallel execution via multiprocessing.

Responsibilities:
    - Worker process initialization (shared locks/counters)
    - Sequential simulation loops with progress tracking
    - Aggregate statistics computation (mean, std dev)
    - JSON logging and result persistence

Architecture:
    - single_simulation: Worker function for one (policy, sample) pair
    - sequential_simulations: Runs multiple samples sequentially
    - init_single_sim_worker: Multiprocessing initializer
    - display_log_metrics: Pretty-prints results to terminal

The simulation execution delegates to SimulationContext (states.py)
which manages the state machine for each run.

Functions:
    init_single_sim_worker: Initialize globals for parallel workers
    single_simulation: Execute one simulation run
    sequential_simulations: Execute batch of runs sequentially
    display_log_metrics: Format and display aggregated results
"""

import os
import statistics

from tqdm import tqdm

from logic.src.utils.definitions import ROOT_DIR, SIM_METRICS
from logic.src.utils.log_utils import log_to_json, output_stats

from .checkpoints import CheckpointError
from .states import SimulationContext

# Create a global variable/placeholder for the lock and counter
_lock = None
_counter = None


def init_single_sim_worker(lock_from_main, counter_from_main):
    """
    Initializes global shared resources for parallel simulation workers.

    This function is called by the multiprocessing Pool during its worker
    initialization phase. It sets process-global variables to ensure all
    simulations can access the shared lock and counter.

    Args:
        lock_from_main: Multiprocessing Lock for file I/O synchronization.
        counter_from_main: Multiprocessing Value for overall progress tracking.
    """
    global _lock
    global _counter
    _lock = lock_from_main
    _counter = counter_from_main


def display_log_metrics(output_dir, size, n_samples, days, area, policies, log, log_std=None, lock=None):
    """
    Pretty-prints aggregated simulation results to the console.

    Formats and displays metrics for multiple policies, including mean values
    and standard deviations if multiple samples were processed.

    Args:
        output_dir: Base directory for results.
        size: Graph size (number of bins).
        n_samples: Number of samples/seeds processed.
        days: Simulation duration in days.
        area: Geographic area name.
        policies: List of policy names.
        log: Dict mapping policy name to mean metrics list.
        log_std: Dict mapping policy name to std dev metrics list (optional).
        lock: Thread/process lock for safe printing/logging.
    """
    if n_samples > 1:
        output_dir = os.path.join(
            ROOT_DIR,
            "assets",
            output_dir,
            str(days) + "_days",
            str(area) + "_" + str(size),
        )
        dit = {}
        std_dit = {}
        for pol, lg, lg_st in zip(policies, log, log_std):
            dit[pol] = lg
            std_dit[pol] = lg_st
        log_to_json(
            os.path.join(output_dir, f"log_mean_{n_samples}N.json"),
            SIM_METRICS,
            log,
            sample_id=None,
            lock=lock,
        )
        log_to_json(
            os.path.join(output_dir, f"log_std_{n_samples}N.json"),
            SIM_METRICS,
            log_std,
            sample_id=None,
            lock=lock,
        )
        for lg, lg_std, pol in zip(log.values(), log_std.values(), log.keys()):
            logm = lg.values() if isinstance(lg, dict) else lg
            logs = lg_std.values() if isinstance(lg_std, dict) else lg_std
            tmp_lg = [(str(x), str(y)) for x, y in zip(logm, logs)]
            print(f"\n{pol} log:")
            for (x, y), key in zip(tmp_lg, SIM_METRICS):
                print(f"- {key} value: {x[: x.find('.') + 3]} +- {y[: y.find('.') + 5]}")
    else:
        for pol, lg in log.items():
            print(f"\n{pol} log:")
            for key, val in zip(SIM_METRICS, lg):
                print(f"- {key}: {val}")
    return


def single_simulation(opts, device, indices, sample_id, pol_id, model_weights_path, n_cores):
    """
    Executes a single simulation run for one (policy, sample) pair.

    Worker function for parallel execution via multiprocessing. Retrieves
    shared locks/counters from globals and delegates to SimulationContext.

    Args:
        opts: Simulation configuration dictionary
        device: torch.device for neural models
        indices: Bin subset indices for this sample
        sample_id: Sample/seed identifier
        pol_id: Policy index in opts['policies']
        model_weights_path: Path to pretrained model weights
        n_cores: Number of CPU cores (for progress bar positioning)

    Returns:
        Result dictionary from SimulationContext.run()
    """
    # Retrieve the shared objects via the global variables initialized by init_worker
    global _lock
    global _counter

    variables_dict = {
        "lock": _lock,
        "counter": _counter,
        "tqdm_pos": os.getpid() % n_cores,
    }

    context = SimulationContext(opts, device, indices, sample_id, pol_id, model_weights_path, variables_dict)
    return context.run()


def sequential_simulations(opts, device, indices_ls, sample_idx_ls, model_weights_path, lock):
    """
    Executes multiple simulation runs sequentially with aggregation.

    Runs all (policy, sample) pairs in sequence, computing mean and standard
    deviation across samples. Displays overall progress bar and handles
    checkpoint errors gracefully.

    Args:
        opts: Simulation configuration dictionary
        device: torch.device for neural models
        indices_ls: List of bin index arrays (one per sample)
        sample_idx_ls: List of sample IDs for each policy
        model_weights_path: Path to pretrained model weights
        lock: Threading lock for file I/O synchronization

    Returns:
        Tuple containing:
            - log: Dict[policy] -> List[mean_metrics]
            - log_std: Dict[policy] -> List[std_metrics] (None if n_samples=1)
            - failed_log: List of error result dictionaries

    Note:
        If opts['resume']=True, loads existing logs instead of recomputing.
    """
    log = {}
    failed_log = []
    if opts["n_samples"] > 1:
        log_std = {}
        log_full = dict.fromkeys(opts["policies"], [])
    else:
        log_std = None

    # Create overall progress bar FIRST with position=1
    overall_progress = tqdm(
        total=sum(len(sublist) for sublist in sample_idx_ls) * opts["days"],
        desc="Overall progress",
        disable=opts["no_progress_bar"],
        position=1,
        leave=True,
    )  # Ensure it stays visible

    # data_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator") # Not needed here
    results_dir = os.path.join(
        ROOT_DIR,
        "assets",
        opts["output_dir"],
        str(opts["days"]) + "_days",
        str(opts["area"]) + "_" + str(opts["size"]),
    )

    for pol_id, policy in enumerate(opts["policies"]):
        for sample_id in sample_idx_ls[pol_id]:
            try:
                variables_dict = {
                    "lock": lock,
                    "overall_progress": overall_progress,
                    "tqdm_pos": 1,
                }

                context = SimulationContext(
                    opts,
                    device,
                    indices_ls[sample_id],
                    sample_id,
                    pol_id,
                    model_weights_path,
                    variables_dict,
                )
                result_dict = context.run()

                # Aggregate execution result
                if result_dict and "success" in result_dict and result_dict["success"]:
                    lg = result_dict[policy]

                    if opts["n_samples"] > 1:
                        log_full[policy].append(lg)
                    else:
                        log[policy] = lg

            except CheckpointError as e:
                failed_log.append(e.error_result)
                pass

            except Exception as e:
                raise e

        if opts["n_samples"] > 1:
            if opts["resume"]:
                log, log_std = output_stats(
                    ROOT_DIR,
                    opts["days"],
                    opts["size"],
                    opts["output_dir"],
                    opts["area"],
                    opts["n_samples"],
                    [policy],
                    SIM_METRICS,
                    lock,
                )
            else:
                log[policy] = [*map(statistics.mean, zip(*log_full[policy]))]
                log_std[policy] = [*map(statistics.stdev, zip(*log_full[policy]))]
                log_to_json(
                    os.path.join(results_dir, f"log_mean_{opts['n_samples']}N.json"),
                    SIM_METRICS,
                    {policy: log[policy]},
                    lock=lock,
                )
                log_to_json(
                    os.path.join(results_dir, f"log_std_{opts['n_samples']}N.json"),
                    SIM_METRICS,
                    {policy: log_std[policy]},
                    lock=lock,
                )

    # Close overall progress bar
    overall_progress.close()
    return log, log_std, failed_log
