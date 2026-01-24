"""
Simulation Testing Pipeline.

This script manages large-scale simulation experiments to evaluate the performance of different
routing policies (Neural, OR-based, Heuristics) over extended periods (e.g., 31 days).
It leverages multiprocessing to run multiple independent simulations in parallel and
reports aggregate metrics like Profit, Cost, and Waste components.
"""

import argparse
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
import traceback
from collections import defaultdict
from multiprocessing.pool import ThreadPool

import numpy as np
import torch
from tqdm import tqdm

import logic.src.utils.definitions as udef
from logic.src.cli import ConfigsParser, add_test_sim_args, validate_test_sim_args
from logic.src.pipeline.simulations.loader import load_indices, load_simulator_data
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


def simulator_testing(opts, data_size, device):
    """
    Orchestrates the parallel execution of multiple simulation runs.

    Handles task allocation (multiprocessing Pool), result collection,
    progress tracking via tqdm, and final logging/GUI reporting.

    Args:
        opts (dict): Validated configuration options.
        data_size (int): Total number of bins in the area.
        device (torch.device): Computation device for neural policies.
    """
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
            print("Simulations left to run:")
            for key, val in sample_idx_dict.items():
                print("- {}: {}".format(key, len(val)))
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

        # Callback to update progress
        def _update_result(result):
            success = result.pop("success")
            if isinstance(result, dict) and success:
                log_tmp[list(result.keys())[0]].append(list(result.values())[0])
                # pbar.update(1)
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

                    # We use a short interruptible wait time and check the status of tasks directly.
                    # We wait on the *first* incomplete task, which is key for responsiveness.
                    first_incomplete = next((task for task in tasks if not task.ready()), None)
                    if first_incomplete:
                        try:
                            # Use get() with a short timeout to make it interruptible
                            first_incomplete.get(timeout=udef.PBAR_WAIT_TIME)
                        except mp.TimeoutError:
                            # Expected when the task isn't finished yet
                            pass
                        except Exception:
                            # Catch any other exception that might have occurred in the worker
                            pass

                # Final update after the loop breaks
                pbar.update(counter.value - last_count)

                # Collect final results/handle exceptions
                for task in tasks:
                    try:
                        task.get()
                    except Exception as e:
                        print(f"Task failed with exception: {e}")
                        traceback.print_exc(file=sys.stdout)
        except KeyboardInterrupt:
            print("\n\n[WARNING] Caught CTRL+C. Terminating worker processes...")
            p.terminate()
            # p.join()
            raise
        finally:
            if "p" in locals() and p is not None:
                try:
                    p.close()
                except ValueError:
                    pass
                try:
                    # This join is only hit on a successful or non-KeyboardInterrupt error exit.
                    p.join()
                except Exception as e:
                    if not isinstance(e, KeyboardInterrupt):
                        raise Exception(f"[CLEANUP ERROR] Failed to join pool cleanly: {type(e).__name__}")
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


def run_wsr_simulator_test(opts):
    """
    Main entry point for the simulation test script.

    Performs initial setup, including:
    1. Seed initialization.
    2. Area-specific bin count determination.
    3. Policy name expansion (e.g., expanding 'regular' to 'regular1', 'regular2').
    4. Output directory preparation.
    5. Dispatching to the simulation testing engine.

    Args:
        opts (dict): Validated configuration options.
    """
    # Set the random seed and execute the program
    random.seed(opts["seed"])
    np.random.seed(opts["seed"])
    torch.manual_seed(opts["seed"])
    # Dynamically determine the available pool size based on area and waste type
    try:
        # Use opts.get("data_dir") which might be None if using default filesystem repository
        data_tmp, _ = load_simulator_data(opts.get("data_dir"), opts["size"], opts["area"], opts["waste_type"])
        data_size = len(data_tmp)
    except Exception:
        # Fallback to legacy hardcoded values if loading fails
        # print(f"DEBUG: load_simulator_data failed with {repr(e)}. Fallback used.")
        if opts["area"] == "mixrmbac" and opts["size"] not in [20, 50, 225]:
            data_size = 20 if opts["size"] < 20 else 50 if opts["size"] < 50 else 225
        elif opts["area"] == "riomaior" and opts["size"] not in [57, 104, 203, 317]:
            data_size = 317
        elif opts["area"] == "both" and opts["size"] not in [57, 371, 485, 542]:
            data_size = 57 if opts["size"] < 57 else 371 if opts["size"] < 371 else 485 if opts["size"] < 485 else 542
        else:
            data_size = opts["size"]

    print(f"Area {opts['area']} ({data_size} available) for {opts['size']} bins")
    if data_size != opts["size"] and (opts["bin_idx_file"] is None or opts["bin_idx_file"] == ""):
        opts["bin_idx_file"] = f"graphs_{opts['size']}V_{opts['n_samples']}N.json"

    # Define the full policy names
    policies = []
    for pol in opts["policies"]:
        if "policy_last_minute_and_path" in pol:
            tmp_pols = [f"policy_last_minute_and_path{cf}" for cf in opts["plastminute_cf"]]
        elif "policy_last_minute" in pol:
            tmp_pols = [f"policy_last_minute{cf}" for cf in opts["plastminute_cf"]]
        elif "policy_regular" in pol:
            tmp_pols = [f"policy_regular{lvl}" for lvl in opts["pregular_level"]]
        elif "gurobi" in pol:
            tmp_pols = [f"gurobi_vrpp{gp}" for gp in opts["gurobi_param"]]
        elif "hexaly" in pol:
            tmp_pols = [f"hexaly_vrpp{hp}" for hp in opts["hexaly_param"]]
        elif "policy_look_ahead" in pol:
            tmp_pols = [pol.replace("ahead", f"ahead_{lac}") for lac in opts["lookahead_configs"]]
        else:
            tmp_pols = [pol]

        for tmp_pol in tmp_pols:
            policies.append("{}_{}".format(tmp_pol, opts["data_distribution"]))

    opts["policies"] = policies
    print("Policy full names:", policies)

    # Setup the output directories
    try:
        parent_dir = os.path.join(
            udef.ROOT_DIR,
            "assets",
            opts["output_dir"],
            f"{opts['days']}_days",
            f"{opts['area']}_{opts['size']}",
        )
        os.makedirs(parent_dir, exist_ok=True)
        os.makedirs(
            os.path.join(parent_dir, "fill_history", opts["data_distribution"]),
            exist_ok=True,
        )
        os.makedirs(os.path.join(udef.ROOT_DIR, opts["checkpoint_dir"]), exist_ok=True)
        os.makedirs(os.path.join(parent_dir, opts["checkpoint_dir"]), exist_ok=True)
    except Exception:
        raise Exception("directories to save WSR simulator test output files do not exist and could not be created")

    # Set the device and run test simulation(s)
    device = torch.device("cpu" if not torch.cuda.is_available() else f"cuda:{torch.cuda.device_count() - 1}")
    try:
        simulator_testing(opts, data_size, device)
    except Exception as e:
        raise Exception(f"failed to execute WSmart+ Route simulations due to {repr(e)}")


if __name__ == "__main__":
    exit_code = 0
    parser = ConfigsParser(
        description="WSmart Route Simulator Test Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_test_sim_args(parser)
    try:
        parsed_args = parser.parse_process_args(sys.argv[1:], "test_sim")
        args = validate_test_sim_args(parsed_args)
        run_wsr_simulator_test(args)
    except (argparse.ArgumentError, AssertionError) as e:
        exit_code = 1
        parser.print_help()
        print(f"Error: {e}", file=sys.stderr)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(str(e), file=sys.stderr)
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)
