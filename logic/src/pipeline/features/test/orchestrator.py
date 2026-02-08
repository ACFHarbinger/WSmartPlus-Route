"""
Simulation Orchestrator.
"""

import multiprocessing as mp
import os
import statistics
import sys
import time
import traceback
from collections import defaultdict
from multiprocessing.pool import ThreadPool

from loguru import logger
from tqdm import tqdm

import logic.src.constants as udef
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
from logic.src.utils.logging.logger_writer import setup_logger_redirection


def simulator_testing(opts, data_size, device):
    """
    Orchestrates the parallel execution of multiple simulation runs.
    """
    setup_logger_redirection()

    manager = mp.Manager()
    lock = manager.Lock()
    sample_idx_dict = {pol: list(range(opts["n_samples"])) for pol in opts["policies"]}
    if opts["resume"]:
        to_remove = runs_per_policy(
            udef.ROOT_DIR,
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
        assert n_cores == 0
        n_cores = task_count if task_count <= mp.cpu_count() - 1 else mp.cpu_count() - 1

    if data_size != opts["size"]:
        indices = load_indices(opts["bin_idx_file"], opts["n_samples"], opts["size"], data_size, lock)
        if len(indices) == 1:
            indices = [indices[0]] * opts["n_samples"]
        assert len(indices) == opts["n_samples"]
    else:
        indices = [None] * opts["n_samples"]

    weights_path = os.path.join(udef.ROOT_DIR, "assets", "model_weights")
    if n_cores > 1:
        udef.update_lock_wait_time(n_cores)
        counter = mp.Value("i", 0)
        if opts["n_samples"] > 1:
            args = [
                (indices[sid], sid, pol_id) for pol_id in range(len(opts["policies"])) for sid in sample_idx_ls[pol_id]
            ]
        else:
            args = [(indices[0], 0, pol_id) for pol_id in range(len(opts["policies"]))]

        def _update_result(result):
            """update result.

            Args:
                    result (Any): Description of result.
            """
            success = result.pop("success")
            if isinstance(result, dict) and success:
                log_tmp[list(result.keys())[0]].append(list(result.values())[0])
            else:
                error_policy = result.get("policy", "unknown")
                error_sample = result.get("sample_id", "unknown")
                error_msg = result.get("error", "Unknown error")
                print(f"Simulation failed: {error_policy} #{error_sample} - {error_msg}")
                failed_log.append(result)

        print(f"Launching {task_count} WSmart Route simulations on {n_cores} CPU cores...")
        max_lock_timeout = time.strftime("%H:%M:%S", time.gmtime(udef.LOCK_TIMEOUT))
        proc_lock_timeout = time.strftime("%H:%M:%S", time.gmtime(udef.LOCK_TIMEOUT // n_cores))
        print(f"- Maximum lock wait time: {max_lock_timeout} ({proc_lock_timeout} per used thread)")
        mp.set_start_method("spawn", force=True)
        p = ThreadPool(
            processes=n_cores,
            initializer=init_single_sim_worker,
            initargs=(
                lock,
                counter,
            ),
        )
        try:
            with tqdm(
                total=len(args) * opts["days"],
                disable=opts["no_progress_bar"],
                position=1,
                desc="Overall progress",
                dynamic_ncols=True,
                colour="black",
            ) as pbar:
                log_tmp = manager.dict()
                failed_log = manager.list()
                for policy in opts["policies"]:
                    log_tmp[policy] = manager.list()

                tasks = []
                for arg_tup in args:
                    task = p.apply_async(
                        single_simulation,
                        args=(opts, device, *arg_tup, weights_path, n_cores),
                        callback=_update_result,
                    )
                    tasks.append(task)

                last_count = 0
                while True:
                    if all(task.ready() for task in tasks):
                        break

                    current_count = counter.value
                    if current_count != last_count:
                        pbar.update(current_count - last_count)
                        last_count = current_count

                    first_incomplete = next((task for task in tasks if not task.ready()), None)
                    if first_incomplete:
                        try:
                            first_incomplete.get(timeout=udef.PBAR_WAIT_TIME)
                        except mp.TimeoutError:
                            pass
                        except Exception:
                            pass

                pbar.update(counter.value - last_count)

                for task in tasks:
                    try:
                        task.get()
                    except Exception as e:
                        print(f"Task failed with exception: {e}")
                        traceback.print_exc(file=sys.stdout)
        except KeyboardInterrupt:
            print("\n\n[WARNING] Caught CTRL+C. Forcing immediate shutdown...")
            sys.stdout.flush()
            sys.stderr.flush()
            try:
                if "p" in locals() and p is not None:
                    p.terminate()
            except Exception:
                pass
            os._exit(1)
        finally:
            if "p" in locals() and p is not None:
                try:
                    p.close()
                except ValueError:
                    pass
        if opts["n_samples"] > 1:
            if opts["resume"]:
                log, log_std = output_stats(
                    udef.ROOT_DIR,
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
                if isinstance(log_tmp, mp.managers.DictProxy):
                    for key, val in log_tmp.items():
                        log_full[key].extend(val)
                else:
                    for result in log_tmp:
                        for key, val in result.items():
                            log_full[key].append(val)

                for pol in opts["policies"]:
                    log[pol] = [*map(statistics.mean, zip(*log_full[pol]))]
                    log_std[pol] = [*map(statistics.stdev, zip(*log_full[pol]))]
        else:
            log = {pol: res[0] for pol, res in log_tmp.items() if res}
            log_std = None
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
