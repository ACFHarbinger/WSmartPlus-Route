"""
Simulation Orchestrator.
"""

import multiprocessing as mp
import os
import signal
import statistics
import sys
import time
import traceback
from collections import defaultdict
from multiprocessing.pool import Pool
from typing import Any, Dict

from loguru import logger

import logic.src.constants as udef
from logic.src.constants import METRICS, SIM_METRICS
from logic.src.pipeline.callbacks.policy_summary import PolicySummaryCallback
from logic.src.pipeline.callbacks.simulation_display import SimulationDisplay
from logic.src.pipeline.simulations.repository import load_indices
from logic.src.pipeline.simulations.simulator import (
    display_log_metrics,
    init_single_sim_worker,
    sequential_simulations,
    single_simulation,
)
from logic.src.utils.logging.log_utils import (
    output_stats,
    runs_per_policy,
    send_final_output_to_gui,
)
from logic.src.utils.logging.logger_writer import LoggerWriter, setup_logger_redirection


def simulator_testing(opts, data_size, device):
    """
    Orchestrates the parallel execution of multiple simulation runs.
    """
    log_file = setup_logger_redirection()

    # Capture original stderr for shutdown messages if redirected
    original_stderr = sys.stderr
    if isinstance(original_stderr, LoggerWriter):
        original_stderr = original_stderr.terminal

    # Display policy summary
    if not opts.get("no_progress_bar"):
        PolicySummaryCallback().display(opts)

    # Register immediate shutdown handler for CTRL+C
    def _shutdown_handler(sig, frame):
        original_stderr.write("\n\n[WARNING] Caught CTRL+C (SIGINT). Forcing immediate shutdown...\n")
        original_stderr.flush()
        # Kill the entire process group to ensure all workers/threads die
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except OSError:
            # Fallback if killpg fails
            os._exit(1)

    signal.signal(signal.SIGINT, _shutdown_handler)

    manager = mp.Manager()
    lock = manager.Lock()
    shared_metrics = manager.dict()
    opts["shared_metrics"] = shared_metrics
    sample_idx_dict = {pol: list(range(opts["n_samples"])) for pol in opts["policies"]}
    if opts["resume"]:
        to_remove = runs_per_policy(  # type: ignore[call-arg, misc]
            udef.ROOT_DIR,  # type: ignore[arg-type]
            opts["days"],
            [opts["size"]],
            opts["output_dir"],
            opts["area"],
            [opts["n_samples"]],
            opts["policies"],
            lock=lock,
        )[0]
        for pol in opts["policies"]:
            if len(to_remove[pol]) > 0:
                sample_idx_dict[pol] = [x for x in sample_idx_dict[pol] if x not in to_remove[pol]]

        sample_idx_ls = [list(val) for val in sample_idx_dict.values()]
        task_count = sum(len(sample_idx) for sample_idx in sample_idx_ls)
        if task_count < sum([opts["n_samples"]] * len(opts["policies"])):
            logger.info("Simulations left to run:")
            for key, val in sample_idx_dict.items():
                logger.info("- {}: {}".format(key, len(val)))
    else:
        sample_idx_ls = [list(val) for val in sample_idx_dict.values()]
        task_count = sum([opts["n_samples"]] * len(opts["policies"]))

    n_cores = opts.get("cpu_cores", 0)
    if n_cores >= 1:
        n_cores = task_count if task_count <= n_cores else n_cores
    else:
        assert n_cores == 0, f"cpu_cores must be non-negative, got {n_cores}"
        n_cores = task_count if task_count <= mp.cpu_count() - 1 else mp.cpu_count() - 1

    if data_size != opts["size"]:
        indices = load_indices(opts["bin_idx_file"], opts["n_samples"], opts["size"], data_size, lock)
        if len(indices) == 1:
            indices = [indices[0]] * opts["n_samples"]
        assert len(indices) == opts["n_samples"], (
            f"Mismatch between loaded indices count ({len(indices)}) and n_samples ({opts['n_samples']}). "
            f"Expected exactly {opts['n_samples']} indices."
        )
    else:
        indices = [None] * opts["n_samples"]

    weights_path = os.path.join(udef.ROOT_DIR, "assets", "model_weights")
    if n_cores > 1:
        log, log_std, failed_log = _run_parallel_simulations(
            opts,
            device,
            indices,
            sample_idx_ls,
            weights_path,
            lock,
            manager,
            n_cores,
            task_count,
            log_file,
            original_stderr,
        )
    else:
        print(f"Launching {task_count} WSmart Route simulations on a single CPU core...")
        log, log_std, failed_log = sequential_simulations(opts, device, indices, sample_idx_ls, weights_path, lock)

    realtime_log_path = os.path.join(
        udef.ROOT_DIR,
        "assets",
        opts["output_dir"],
        str(opts["days"]) + "_days",
        str(opts["area"]) + "_" + str(opts["size"]),
        f"log_realtime_{opts['data_distribution']}_{opts['n_samples']}N.jsonl",
    )
    send_final_output_to_gui(log, log_std, opts["n_samples"], opts["policies"], realtime_log_path)
    if not opts.get("no_progress_bar"):
        display_log_metrics(
            opts["output_dir"],
            opts["size"],
            opts["n_samples"],
            opts["days"],
            opts["area"],
            opts["policies"],
            log,
            log_std,
            lock,
        )


def _process_display_updates(
    display: SimulationDisplay,
    shared_metrics: dict,
    log_tmp: dict,
    last_reported_days: dict,
    opts: dict,
    loop_tic: float,
    counter: Any,
):
    """
    Process real-time simulation metrics and update the dashboard display.

    This logic involves aggregating metrics from both active running samples
    (via shared_metrics) and completed samples (via log_tmp), calculating
    averages and standard deviations, and triggering a display refresh.
    """
    policy_updates = {}
    new_daily_data = []  # For chart updates

    policy_days_done: Dict[str, int] = defaultdict(int)
    policy_sample_metrics: Dict[str, Dict[str, list]] = defaultdict(lambda: {k: [] for k in SIM_METRICS})  # type: ignore[assignment]
    policy_sample_counts: Dict[str, int] = defaultdict(int)

    # A. Process ACTIVE samples from shared_metrics
    # shared_metrics now contains CUMULATIVE totals from RunningState
    for _key, data in shared_metrics.items():
        pol = data["policy"]
        sid = data["sample_id"]
        day = data["day"]
        metrics = data["metrics"]

        policy_days_done[pol] += day
        policy_sample_counts[pol] += 1

        # Use METRICS constant to filter numeric values correctly
        for k in METRICS:
            if k in metrics:
                policy_sample_metrics[pol][k].append(metrics[k])
        policy_sample_metrics[pol]["days"].append(day)

        # Report to chart if it's a new day
        if (pol, sid) not in last_reported_days or last_reported_days[(pol, sid)] < day:
            # Use daily_delta for chart to keep it showing "per day" performance
            delta = data.get("daily_delta", metrics)
            new_daily_data.append({"policy": pol, "day": day, "metrics": delta})
            last_reported_days[(pol, sid)] = day

    # B. Add COMPLETED samples from log_tmp
    for pol, results in log_tmp.items():
        for res in results:
            # res is a list of final totals [ovr, kg, ncol, ..., days, time]
            policy_sample_counts[pol] += 1
            policy_days_done[pol] += res[8]  # days index
            for i, k in enumerate(SIM_METRICS):
                policy_sample_metrics[pol][k].append(res[i])

    # 3. Calculate averages and update display
    elapsed_total = time.time() - loop_tic
    for pol in opts["policies"]:
        n_finished = len(log_tmp[pol])
        divisor = max(1, policy_sample_counts[pol])

        avg_metrics = {}
        for k in SIM_METRICS:
            vals = policy_sample_metrics[pol][k]
            if not vals:
                avg_metrics[k] = (0.0, 0.0)
                continue

            if k == "time" and sum(vals) == 0:
                m = elapsed_total / divisor
                s = 0.0
            else:
                m = statistics.mean(vals)
                s = statistics.stdev(vals) if len(vals) > 1 else 0.0

            avg_metrics[k] = (m, s)

        policy_updates[pol] = {
            "metrics": avg_metrics,
            "completed": n_finished,
            "total_days_done": policy_days_done[pol],
        }

    display.update(counter.value, policy_updates, new_daily_data)


def _run_parallel_simulations(
    opts,
    device,
    indices,
    sample_idx_ls,
    weights_path,
    lock,
    manager,
    n_cores,
    task_count,
    log_file,
    original_stderr,
):
    """
    Execute simulations in parallel using a multiprocessing pool.

    Coordinates the parallel execution workflow:
    1. Prepares task arguments and result callback
    2. Creates and configures multiprocessing pool
    3. Submits all simulation tasks
    4. Monitors progress with optional dashboard display
    5. Aggregates results after completion

    Args:
        opts (dict): Configuration options
        device (torch.device): Target device for neural models
        indices (list): Bin indices for each sample
        sample_idx_ls (list): Sample indices per policy
        weights_path (str): Path to model weights directory
        lock (mp.Lock): Synchronization lock
        manager (mp.Manager): Multiprocessing manager
        n_cores (int): Number of CPU cores to use
        task_count (int): Total number of tasks
        log_file (str): Path to log file
        original_stderr (TextIO): Original stderr stream for error handling

    Returns:
        tuple: (log, log_std, failed_log) - aggregated results, standard deviations, and failed tasks
    """
    udef.update_lock_wait_time(n_cores)
    counter = mp.Value("i", 0)

    # Prepare task arguments
    args = _prepare_parallel_task_args(opts, indices, sample_idx_ls)

    # Print execution info
    _print_execution_info(opts, task_count, n_cores)

    # Create multiprocessing pool
    mp.set_start_method("spawn", force=True)
    pool = Pool(
        processes=n_cores,
        initializer=init_single_sim_worker,
        initargs=(lock, counter, opts["shared_metrics"], log_file),
    )

    try:
        log, log_std, failed_log = _execute_and_monitor_tasks(
            pool, opts, device, args, weights_path, n_cores, counter, manager, lock
        )
        return log, log_std, failed_log

    except KeyboardInterrupt:
        _handle_shutdown(pool, original_stderr)

    finally:
        _cleanup_multiprocessing_pool(pool)


def _prepare_parallel_task_args(opts, indices, sample_idx_ls):
    """
    Prepare argument tuples for parallel task execution.

    Args:
        opts (dict): Configuration options
        indices (list): Bin indices for each sample
        sample_idx_ls (list): Sample indices per policy

    Returns:
        list: List of argument tuples (index, sample_id, policy_id)
    """
    if opts["n_samples"] > 1:
        return [(indices[sid], sid, pol_id) for pol_id in range(len(opts["policies"])) for sid in sample_idx_ls[pol_id]]
    else:
        return [(indices[0], 0, pol_id) for pol_id in range(len(opts["policies"]))]


def _print_execution_info(opts, task_count, n_cores):
    """
    Print information about parallel execution configuration.

    Args:
        task_count (int): Total number of tasks to execute
        n_cores (int): Number of CPU cores to use
    """
    if not opts.get("no_progress_bar"):
        print(f"Launching {task_count} WSmart Route simulations on {n_cores} CPU cores...")
        max_lock_timeout = time.strftime("%H:%M:%S", time.gmtime(udef.LOCK_TIMEOUT))
        proc_lock_timeout = time.strftime("%H:%M:%S", time.gmtime(udef.LOCK_TIMEOUT // n_cores))
        print(f"[INFO] Maximum lock wait time: {max_lock_timeout} ({proc_lock_timeout} per used thread)")


def _execute_and_monitor_tasks(pool, opts, device, args, weights_path, n_cores, counter, manager, lock):
    """
    Execute parallel tasks and monitor their progress with optional dashboard display.

    Args:
        pool (Pool): Multiprocessing pool
        opts (dict): Configuration options
        device (torch.device): Target device
        args (list): Task argument tuples
        weights_path (str): Path to model weights
        n_cores (int): Number of cores
        counter (mp.Value): Shared counter for progress tracking
        manager (mp.Manager): Multiprocessing manager
        lock (mp.Lock): Synchronization lock

    Returns:
        tuple: (log, log_std, failed_log)
    """
    # Initialize display and result containers
    # Check both root and sim level no_progress_bar
    no_pbar = opts.get("no_progress_bar", False)
    display = _initialize_simulation_display(opts) if not no_pbar else None
    log_tmp, failed_log = _create_result_containers(manager, opts)

    # Create result callback closure
    def _update_result(result):
        """Handle task completion results."""
        _process_task_result(result, log_tmp, failed_log)

    # Submit all tasks to the pool
    tasks = _submit_simulation_tasks(pool, opts, device, args, weights_path, n_cores, _update_result)

    # Monitor tasks until completion
    _monitor_tasks_until_complete(tasks, display, opts, counter, log_tmp)

    # Stop display and collect final results
    if display:
        display.stop()

    _collect_all_task_results(tasks)

    # Aggregate and return results
    log, log_std = _aggregate_final_results(log_tmp, opts, lock)
    return log, log_std, failed_log


def _initialize_simulation_display(opts):
    """
    Initialize the simulation dashboard display if enabled.

    Args:
        opts (dict): Configuration options

    Returns:
        SimulationDisplay or None: Display instance or None if disabled
    """
    if opts.get("no_progress_bar"):
        return None

    display = SimulationDisplay(policies=opts["policies"], n_samples=opts["n_samples"], total_days=opts["days"])
    display.start()
    return display


def _create_result_containers(manager, opts):
    """
    Create shared containers for storing simulation results.

    Args:
        manager (mp.Manager): Multiprocessing manager
        opts (dict): Configuration options

    Returns:
        tuple: (log_tmp, failed_log) - result dictionaries and failed task list
    """
    log_tmp = manager.dict()
    failed_log = manager.list()

    for policy in opts["policies"]:
        log_tmp[policy] = manager.list()

    return log_tmp, failed_log


def _process_task_result(result, log_tmp, failed_log):
    """
    Process a single task completion result.

    Args:
        result (dict): Task result dictionary
        log_tmp (dict): Temporary log storage
        failed_log (list): List of failed tasks
    """
    success = result.pop("success")

    if not (isinstance(result, dict) and success):
        error_policy = result.get("policy", "unknown")
        error_sample = result.get("sample_id", "unknown")
        error_msg = result.get("error", "Unknown error")
        print(f"Simulation failed: {error_policy} #{error_sample} - {error_msg}")
        failed_log.append(result)
        return

    log_tmp[list(result.keys())[0]].append(list(result.values())[0])


def _submit_simulation_tasks(pool, opts, device, args, weights_path, n_cores, callback):
    """
    Submit all simulation tasks to the multiprocessing pool.

    Args:
        pool (Pool): Multiprocessing pool
        opts (dict): Configuration options
        device (torch.device): Target device
        args (list): Task argument tuples
        weights_path (str): Path to model weights
        n_cores (int): Number of cores
        callback (callable): Result callback function

    Returns:
        list: List of AsyncResult objects
    """
    tasks = []
    for arg_tup in args:
        task = pool.apply_async(
            single_simulation,
            args=(opts, device, *arg_tup, weights_path, n_cores),
            callback=callback,
        )
        tasks.append(task)

    return tasks


def _monitor_tasks_until_complete(tasks, display, opts, counter, log_tmp):
    """
    Monitor task progress and update display until all tasks complete.

    Args:
        tasks (list): List of AsyncResult objects
        display (SimulationDisplay or None): Display instance
        opts (dict): Configuration options
        counter (mp.Value): Shared counter
        log_tmp (dict): Temporary log storage
    """
    last_reported_days = {}  # type: ignore[var-annotated]
    loop_tic = time.time()

    while not all(task.ready() for task in tasks):
        if display:
            _process_display_updates(
                display=display,
                shared_metrics=opts["shared_metrics"],
                log_tmp=log_tmp,
                last_reported_days=last_reported_days,
                opts=opts,
                loop_tic=loop_tic,
                counter=counter,
            )
        time.sleep(udef.PBAR_WAIT_TIME)


def _collect_all_task_results(tasks):
    """
    Collect results from all tasks, logging any exceptions.

    Args:
        tasks (list): List of AsyncResult objects
    """
    for task in tasks:
        try:
            task.get()
        except Exception as e:
            print(f"Task failed with exception: {e}")
            traceback.print_exc(file=sys.stdout)


def _handle_shutdown(pool, original_stderr):
    """
    Handle graceful shutdown on keyboard interrupt.

    Args:
        pool (Pool): Multiprocessing pool
        original_stderr (TextIO): Original stderr stream
    """
    original_stderr.write("\n\n[WARNING] Caught CTRL+C. Forcing immediate shutdown...\n")
    original_stderr.flush()

    try:
        pool.terminate()
        pool.join()
    except Exception:
        pass

    os._exit(1)


def _cleanup_multiprocessing_pool(pool):
    """
    Safely close and join the multiprocessing pool.

    Args:
        pool (Pool): Multiprocessing pool to clean up
    """
    try:
        pool.close()
        pool.join()
    except Exception:
        pass


def _aggregate_final_results(log_tmp, opts, lock):
    """
    Aggregate results from all finished simulation samples.
    """
    if opts["n_samples"] > 1:
        if opts["resume"]:
            return output_stats(  # type: ignore[call-arg, misc]
                udef.ROOT_DIR,  # type: ignore[arg-type]
                opts["days"],
                opts["size"],
                opts["output_dir"],
                opts["area"],
                opts["n_samples"],
                opts["policies"],
                udef.SIM_METRICS,
                lock=lock,
            )
        else:
            log = {}
            log_std = {}
            log_full = defaultdict(list)

            # Extract list from Manager objects
            # Handling both DictProxy and nested results
            for key, val in log_tmp.items():
                log_full[key].extend(val)

            for pol in opts["policies"]:
                if log_full[pol]:
                    log[pol] = [statistics.mean(v) for v in zip(*log_full[pol])]
                    log_std[pol] = [statistics.stdev(v) if len(log_full[pol]) > 1 else 0.0 for v in zip(*log_full[pol])]
                else:
                    log[pol] = [0.0] * len(udef.SIM_METRICS)
                    log_std[pol] = [0.0] * len(udef.SIM_METRICS)
            return log, log_std
    else:
        log = {pol: res[0] for pol, res in log_tmp.items() if res}
        return log, None
