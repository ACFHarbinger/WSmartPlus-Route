import multiprocessing as mp
import os
from multiprocessing.pool import Pool

import logic.src.constants as udef
from logic.src.pipeline.features.test.orchestrator.monitor import (
    collect_all_task_results,
    initialize_simulation_display,
    monitor_tasks_until_complete,
)
from logic.src.pipeline.features.test.orchestrator.results_handler import aggregate_final_results
from logic.src.pipeline.features.test.orchestrator.utils import (
    prepare_parallel_task_args,
    print_execution_info,
)
from logic.src.pipeline.simulations.simulator import (
    init_single_sim_worker,
    single_simulation,
)


def run_parallel_simulations(
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
    """
    udef.update_lock_wait_time(n_cores)
    counter = mp.Value("i", 0)

    # Prepare task arguments
    args = prepare_parallel_task_args(opts, indices, sample_idx_ls)

    # Print execution info
    print_execution_info(opts, task_count, n_cores)

    # Create multiprocessing pool
    mp.set_start_method("spawn", force=True)
    pool = Pool(
        processes=n_cores,
        initializer=init_single_sim_worker,
        initargs=(lock, counter, opts["shared_metrics"], log_file),
    )

    try:
        log, log_std, failed_log = execute_and_monitor_tasks(
            pool, opts, device, args, weights_path, n_cores, counter, manager, lock
        )
        return log, log_std, failed_log

    except KeyboardInterrupt:
        handle_shutdown(pool, original_stderr)

    finally:
        cleanup_multiprocessing_pool(pool)


def execute_and_monitor_tasks(pool, opts, device, args, weights_path, n_cores, counter, manager, lock):
    """
    Execute parallel tasks and monitor their progress.
    """
    no_pbar = opts.get("no_progress_bar", False)
    display = initialize_simulation_display(opts) if not no_pbar else None

    log_tmp = manager.dict()
    failed_log = manager.list()
    for policy in opts["policies"]:
        log_tmp[policy] = manager.list()

    def _update_result(result):
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
            args=(opts, device, *arg_tup, weights_path, n_cores),
            callback=_update_result,
        )
        tasks.append(task)

    monitor_tasks_until_complete(tasks, display, opts, counter, log_tmp)

    if display:
        display.stop()

    collect_all_task_results(tasks)

    log, log_std = aggregate_final_results(log_tmp, opts, lock)
    return log, log_std, failed_log


def handle_shutdown(pool, original_stderr):
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


def cleanup_multiprocessing_pool(pool):
    """
    Safely close and join the multiprocessing pool.
    """
    try:
        pool.close()
        pool.join()
    except Exception:
        pass
