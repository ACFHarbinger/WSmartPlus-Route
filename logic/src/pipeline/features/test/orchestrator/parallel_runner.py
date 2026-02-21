import multiprocessing as mp
import os
from multiprocessing.pool import Pool
from typing import Any, Dict, List, Optional, Tuple

import logic.src.constants as udef
from logic.src.configs import Config
from logic.src.pipeline.features.test.orchestrator.monitor import (
    collect_all_task_results,
    initialize_simulation_display,
    monitor_tasks_until_complete,
)
from logic.src.pipeline.features.test.orchestrator.results_handler import aggregate_final_results
from logic.src.pipeline.simulations.simulator import (
    init_single_sim_worker,
    single_simulation,
)
from logic.src.utils.tasks.simulation_utils import (
    prepare_parallel_task_args,
    print_execution_info,
)


def run_parallel_simulations(
    cfg: Config,
    device: Any,
    indices: List[Any],
    sample_idx_ls: List[List[int]],
    weights_path: str,
    lock: Any,
    manager: Any,
    n_cores: int,
    task_count: int,
    log_file: Optional[str],
    original_stderr: Any,
    shared_metrics: Any,
    tracking_uri: Optional[str] = None,
    tracking_run_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Execute simulations in parallel using a multiprocessing pool.

    Args:
        cfg: Root configuration with ``cfg.sim`` containing simulation params.
        device: Torch device.
        indices: Bin subset indices per sample.
        sample_idx_ls: Sample index lists per policy.
        weights_path: Path to model weights.
        lock: Multiprocessing lock.
        manager: Multiprocessing Manager.
        n_cores: Number of cores to use.
        task_count: Total number of tasks.
        log_file: Path to log file for worker redirection.
        original_stderr: Original stderr for shutdown messages.
        shared_metrics: Multiprocessing Manager.dict for real-time stats.
        tracking_uri: Directory of the WSTracker database (passed to workers).
        tracking_run_id: Parent run UUID that workers should attach to.
    """
    sim = cfg.sim
    udef.update_lock_wait_time(n_cores)
    counter = mp.Value("i", 0)

    # Prepare task arguments
    args = prepare_parallel_task_args(sim.full_policies, sim.n_samples, indices, sample_idx_ls)

    # Print execution info
    print_execution_info(task_count, n_cores)

    # Create multiprocessing pool
    mp.set_start_method("spawn", force=True)
    pool = Pool(
        processes=n_cores,
        initializer=init_single_sim_worker,
        initargs=(lock, counter, shared_metrics, log_file, cfg, tracking_uri, tracking_run_id),
    )

    try:
        log, log_std, failed_log = execute_and_monitor_tasks(
            pool, cfg, device, args, weights_path, n_cores, counter, manager, lock, shared_metrics
        )
        return log, log_std, failed_log

    except KeyboardInterrupt:
        handle_shutdown(pool, original_stderr)
        return {}, None, []  # unreachable but satisfies type checker

    finally:
        cleanup_multiprocessing_pool(pool)


def execute_and_monitor_tasks(
    pool: Pool,
    cfg: Config,
    device: Any,
    args: List[Tuple[Any, ...]],
    weights_path: str,
    n_cores: int,
    counter: Any,
    manager: Any,
    lock: Any,
    shared_metrics: Any,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Execute parallel tasks and monitor their progress.
    """
    sim = cfg.sim
    policies = sim.full_policies
    no_pbar = sim.no_progress_bar
    display = initialize_simulation_display(policies, sim.n_samples, sim.days) if not no_pbar else None

    log_tmp = manager.dict()
    failed_log = manager.list()
    for policy in policies:
        log_tmp[policy] = manager.list()

    def _update_result(result: Dict[str, Any]) -> None:
        success = result.pop("success")
        if not (isinstance(result, dict) and success):
            error_policy = result.get("policy", "unknown")
            error_sample = result.get("sample_id", "unknown")
            error_msg = result.get("error", "Unknown error")
            print(f"Simulation failed: {error_policy} #{error_sample} - {error_msg}")
            failed_log.append(result)
            return
        log_tmp[list(result.keys())[0]].append(list(result.values())[0])

    tasks = []
    for arg_tup in args:
        task = pool.apply_async(
            single_simulation,
            args=(cfg, device, *arg_tup, weights_path, n_cores),
            callback=_update_result,
        )
        tasks.append(task)

    monitor_tasks_until_complete(tasks, display, policies, shared_metrics, counter, log_tmp)

    if display:
        display.stop()

    collect_all_task_results(tasks)

    log, log_std = aggregate_final_results(log_tmp, cfg, lock)
    return log, log_std, failed_log


def handle_shutdown(pool: Pool, original_stderr: Any) -> None:
    """
    Handle graceful shutdown on keyboard interrupt.
    """
    original_stderr.write("\n\n[WARNING] Caught CTRL+C. Forcing immediate shutdown...\n")
    original_stderr.flush()

    try:
        pool.terminate()
        pool.join()
    except Exception:
        pass

    os._exit(1)


def cleanup_multiprocessing_pool(pool: Pool) -> None:
    """
    Safely close and join the multiprocessing pool.
    """
    try:
        pool.close()
        pool.join()
    except Exception:
        pass
