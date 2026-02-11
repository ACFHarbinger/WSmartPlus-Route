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

from __future__ import annotations

import os
import statistics
import sys
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from logic.src.constants import ROOT_DIR, SIM_METRICS
from logic.src.interfaces import ITraversable
from logic.src.utils.logging.log_utils import log_to_json, output_stats

from .checkpoints import CheckpointError
from .states import SimulationContext

if TYPE_CHECKING:
    pass

# Create global variables/placeholders for parallel workers
_lock: Optional[Any] = None
_counter: Optional[Any] = None
_shared_metrics: Optional[Any] = None


def init_single_sim_worker(
    lock_from_main: Any,
    counter_from_main: Any,
    shared_metrics_from_main: Any = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Initializes global shared resources for parallel simulation workers.

    This function is called by the multiprocessing Pool during its worker
    initialization phase. It sets process-global variables to ensure all
    simulations can access the shared resources.

    Args:
        lock_from_main: Multiprocessing Lock for file I/O synchronization.
        counter_from_main: Multiprocessing Value for overall progress tracking.
        shared_metrics_from_main: Multiprocessing Manager.dict for real-time stats.
        log_file: Path to the log file for redirection.
    """
    global _lock
    global _counter
    global _shared_metrics
    _lock = lock_from_main
    _counter = counter_from_main
    _shared_metrics = shared_metrics_from_main

    # Setup logger redirection in the worker process (silent=True to avoid garbling dashboard)
    if log_file:
        from logic.src.utils.logging.logger_writer import setup_logger_redirection

        setup_logger_redirection(log_file, silent=True)


def display_log_metrics(
    output_dir: str,
    size: int,
    n_samples: int,
    days: int,
    area: str,
    policies: List[str],
    log: Dict[str, Union[List[float], Dict[str, float]]],
    log_std: Optional[Dict[str, Union[List[float], Dict[str, float]]]] = None,
    lock: Optional[Any] = None,
) -> None:
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
    if n_samples > 1 and log_std is not None:
        abs_output_dir = os.path.join(
            ROOT_DIR,
            "assets",
            output_dir,
            str(days) + "_days",
            str(area) + "_" + str(size),
        )
        log_to_json(
            os.path.join(abs_output_dir, f"log_mean_{n_samples}N.json"),
            SIM_METRICS,
            log,
            sample_id=None,
            lock=lock,
        )
        log_to_json(
            os.path.join(abs_output_dir, f"log_std_{n_samples}N.json"),
            SIM_METRICS,
            log_std,
            sample_id=None,
            lock=lock,
        )
        for lg, lg_std, pol in zip(log.values(), log_std.values(), log.keys()):
            logm = lg.values() if isinstance(lg, ITraversable) else lg
            logs = lg_std.values() if isinstance(lg_std, ITraversable) else lg_std
            tmp_lg = [(str(x), str(y)) for x, y in zip(logm, logs)]
            print(f"\n{pol} log:")
            for (x, y), key in zip(tmp_lg, SIM_METRICS):
                print(f"- {key} value: {x[: x.find('.') + 3]} +- {y[: y.find('.') + 5]}")
    else:
        for pol, lg in log.items():
            print(f"\n{pol} log:")
            lg_vals = lg.values() if isinstance(lg, ITraversable) else lg
            for key, val in zip(SIM_METRICS, lg_vals):
                print(f"- {key}: {val}")


def single_simulation(
    opts: Dict[str, Any],
    device: torch.device,
    indices: Any,
    sample_id: int,
    pol_id: int,
    model_weights_path: str,
    n_cores: int,
) -> Dict[str, Any]:
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
    global _shared_metrics

    try:
        sys.stdout.flush()
        variables_dict: Dict[str, Any] = {
            "lock": _lock,
            "counter": _counter,
            "shared_metrics": _shared_metrics,
            "tqdm_pos": os.getpid() % n_cores,
        }

        context = SimulationContext(opts, device, indices, sample_id, pol_id, model_weights_path, variables_dict)
        res: Optional[Dict[str, Any]] = context.run()
        return res if res is not None else {}
    except BaseException as e:
        # Force print to real stderr to bypass any redirection
        err_stream = sys.__stderr__ or sys.stderr
        print(
            f"CRITICAL ERROR (BaseException) in single_simulation (Sample {sample_id}, Policy {pol_id}): {e}",
            file=err_stream,
        )
        traceback.print_exc(file=err_stream)
        if err_stream:
            err_stream.flush()
        return {"policy": "unknown", "sample_id": sample_id, "error": str(e), "success": False}


def sequential_simulations(  # noqa: C901
    opts: Dict[str, Any],
    device: torch.device,
    indices_ls: List[Any],
    sample_idx_ls: List[List[int]],
    model_weights_path: str,
    lock: Optional[Any],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
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
    log: Dict[str, Any] = {}
    failed_log: List[Dict[str, Any]] = []
    log_std: Optional[Dict[str, Any]] = None
    log_full: Dict[str, List[List[float]]] = {}

    # Always initialize accumulation structures to support metrics for N=1 case
    log_std = {}
    log_full = {policy: [] for policy in opts["policies"]}

    # Create overall progress bar FIRST with position=1
    overall_progress = tqdm(
        total=sum(len(sublist) for sublist in sample_idx_ls) * opts["days"],
        desc="Overall progress",
        disable=opts["no_progress_bar"],
        position=1,
        leave=True,
    )  # Ensure it stays visible

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
                variables_dict: Dict[str, Any] = {
                    "lock": lock,
                    "overall_progress": overall_progress,
                    "shared_metrics": opts.get("shared_metrics"),
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

                    # Always append to log_full to support uniform aggregation logic
                    log_full[policy].append(lg)
                    log[policy] = lg
            except BaseException as e:
                # Force print to real stderr to bypass any redirection
                err_stream = sys.__stderr__ or sys.stderr
                print(
                    f"CRITICAL ERROR (BaseException) in sequential_simulations (Sample {sample_id}, Policy {pol_id}): {e}",
                    file=err_stream,
                )
                traceback.print_exc(file=err_stream)
                if err_stream:
                    err_stream.flush()
                # Continue with next sample/policy instead of crashing the whole process
                pass
            except CheckpointError:
                # Skip broken checkpoints
                pass

        if opts["n_samples"] >= 1:
            if opts["resume"]:
                res_log, res_std = output_stats(
                    results_dir,
                    opts["n_samples"],
                    [policy],
                    SIM_METRICS,
                    lock=lock,
                )
                if res_log:
                    log.update(res_log)
                if res_std and log_std is not None:
                    log_std.update(res_std)
            elif policy in log_full and log_full[policy]:
                log[policy] = [*map(statistics.mean, zip(*log_full[policy]))]
                if len(log_full[policy]) > 1:
                    if log_std is not None:
                        log_std[policy] = [*map(statistics.stdev, zip(*log_full[policy]))]
                elif log_std is not None:
                    log_std[policy] = [0.0] * len(log[policy])

                log_to_json(
                    os.path.join(results_dir, f"log_mean_{opts['n_samples']}N.json"),
                    SIM_METRICS,
                    {policy: log[policy]},
                    lock=lock,
                )
                if log_std is not None:
                    log_to_json(
                        os.path.join(results_dir, f"log_std_{opts['n_samples']}N.json"),
                        SIM_METRICS,
                        {policy: log_std[policy]},
                        lock=lock,
                    )

    # Close overall progress bar
    overall_progress.close()
    return log, log_std, failed_log
