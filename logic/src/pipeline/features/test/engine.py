"""
Simulation testing engine for WSmart-Route.
"""

import copy
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
from loguru import logger
from tqdm import tqdm

import logic.src.constants as udef
from logic.src.pipeline.simulations.repository import load_indices, load_simulator_data
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


def run_wsr_simulator_test(opts):
    """Main entry point for the simulation test script."""
    random.seed(opts["seed"])
    np.random.seed(opts["seed"])
    torch.manual_seed(opts["seed"])
    try:
        data_tmp, _ = load_simulator_data(opts.get("data_dir"), opts["size"], opts["area"], opts["waste_type"])
        data_size = len(data_tmp)
    except Exception:
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

    policies = []
    from logic.src.utils.configs.config_loader import load_config

    for pol in opts["policies"]:
        tmp_pols = [pol]
        for tmp_pol in tmp_pols:
            prefix_str = ""
            suffix_str = ""
            variant_name = None
            cfg_path = None
            variants = [("", "", None)]

            try:
                cfg_path = os.path.join(udef.ROOT_DIR, "assets", "configs", "policies", f"{tmp_pol}.yaml")
                if not os.path.exists(cfg_path):
                    if not tmp_pol.startswith("policy_"):
                        cfg_path = os.path.join(
                            udef.ROOT_DIR, "assets", "configs", "policies", f"policy_{tmp_pol}.yaml"
                        )

                if os.path.exists(cfg_path):
                    pol_cfg = load_config(cfg_path)
                    inner_cfg = []
                    if pol_cfg:
                        for k, v in pol_cfg.items():
                            if isinstance(v, list):
                                inner_cfg = v
                                break
                            if isinstance(v, dict):
                                if "must_go" in v or "post_processing" in v:
                                    pass
                                else:
                                    for sub_k, sub_v in v.items():
                                        if isinstance(sub_v, list):
                                            inner_cfg = sub_v
                                            variant_name = sub_k
                                            break

                    mg_list = []
                    pp_list = []
                    match_item_idx = -1
                    for idx, item in enumerate(inner_cfg):
                        if isinstance(item, dict):
                            if "must_go" in item:
                                mg_list = item["must_go"]
                                match_item_idx = idx
                            if "post_processing" in item:
                                pp_list = item["post_processing"]

                    variants = []
                    if mg_list and len(mg_list) > 1:
                        for mg_item in mg_list:
                            v_prefix = ""
                            if isinstance(mg_item, str):
                                clean_mg = mg_item.replace("mg_", "").replace(".xml", "").replace(".yaml", "")
                                v_prefix = f"{clean_mg}_"

                            v_suffix = ""
                            if pp_list:
                                first_pp = pp_list[0]
                                if isinstance(first_pp, str):
                                    clean_pp = first_pp.replace("pp_", "").replace(".xml", "").replace(".yaml", "")
                                    v_suffix = f"_{clean_pp}"

                            var_cfg = copy.deepcopy(pol_cfg)
                            var_inner = []
                            if var_cfg:
                                found = False
                                for k, v in var_cfg.items():
                                    if isinstance(v, list):
                                        var_inner = v
                                        found = True
                                        break
                                    if isinstance(v, dict):
                                        if "must_go" in v or "post_processing" in v:
                                            pass
                                        else:
                                            for sub_k, sub_v in v.items():
                                                if isinstance(sub_v, list):
                                                    var_inner = sub_v
                                                    found = True
                                                    break
                                    if found:
                                        break

                            if var_inner and match_item_idx >= 0 and match_item_idx < len(var_inner):
                                if isinstance(var_inner[match_item_idx], dict):
                                    var_inner[match_item_idx]["must_go"] = [mg_item]

                            variants.append((v_prefix, v_suffix, var_cfg))
                    else:
                        prefix_str = ""
                        if mg_list:
                            first_mg = mg_list[0]
                            if isinstance(first_mg, str):
                                clean_mg = first_mg.replace("mg_", "").replace(".xml", "").replace(".yaml", "")
                                prefix_str = f"{clean_mg}_"

                        suffix_str = ""
                        if pp_list:
                            first_pp = pp_list[0]
                            if isinstance(first_pp, str):
                                clean_pp = first_pp.replace("pp_", "").replace(".xml", "").replace(".yaml", "")
                                suffix_str = f"_{clean_pp}"

                        variants.append((prefix_str, suffix_str, None))

            except Exception as e:
                print(f"Warning: Could not load config for naming {tmp_pol}: {e}")
                variants = [("", "", None)]

            for prefix, suffix, custom_cfg in variants:
                middle_name = tmp_pol.replace("policy_", "")
                if variant_name and variant_name.lower() != "default":
                    middle_name = f"{middle_name}_{variant_name}"

                full_name = f"{prefix}{middle_name}{suffix}_{opts['data_distribution']}"
                policies.append(full_name)

                if "config_path" not in opts or not isinstance(opts["config_path"], dict):
                    opts["config_path"] = (
                        opts.get("config_path", {}) if isinstance(opts.get("config_path"), dict) else {}
                    )

                key = full_name
                if custom_cfg:
                    opts["config_path"][key] = custom_cfg
                elif cfg_path and os.path.exists(cfg_path):
                    opts["config_path"][key] = cfg_path

    opts["policies"] = policies

    try:
        parent_dir = os.path.join(
            udef.ROOT_DIR, "assets", opts["output_dir"], f"{opts['days']}_days", f"{opts['area']}_{opts['size']}"
        )
        os.makedirs(parent_dir, exist_ok=True)
        os.makedirs(os.path.join(parent_dir, "fill_history", opts["data_distribution"]), exist_ok=True)
        os.makedirs(os.path.join(udef.ROOT_DIR, opts["checkpoint_dir"]), exist_ok=True)
        os.makedirs(os.path.join(parent_dir, opts["checkpoint_dir"]), exist_ok=True)
    except Exception:
        raise Exception("directories to save WSR simulator test output files do not exist and could not be created")

    device = torch.device("cpu" if not torch.cuda.is_available() else f"cuda:{torch.cuda.device_count() - 1}")
    try:
        simulator_testing(opts, data_size, device)
    except Exception as e:
        raise Exception(f"failed to execute WSmart+ Route simulations due to {repr(e)}")
