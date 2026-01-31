"""
Logging systems for terminal, file, and GUI communication.
"""

from __future__ import annotations

import datetime
import json
import os
import pickle
import statistics
import threading
from collections import Counter
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger

import logic.src.constants as udef
from logic.src.utils.io.files import compose_dirpath, read_json


def setup_system_logger(log_path: str = "logs/system.log", level: str = "INFO") -> Any:
    """
    Configures loguru to log to both console and a file.
    """
    import sys

    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level=level)
    logger.add(log_path, rotation="10 MB", level=level)
    return logger


def log_values(
    cost: torch.Tensor,
    grad_norms: Tuple[torch.Tensor, ...],
    epoch: int,
    batch_id: int,
    step: int,
    l_dict: Dict[str, torch.Tensor],
    tb_logger: Any,
    opts: Dict[str, Any],
) -> None:
    """
    Logs training metrics to the console, TensorBoard, and WandB.

    Args:
        cost: The cost tensor.
        grad_norms: Tuple of gradient norms (all, clipped).
        epoch: Current epoch number.
        batch_id: Current batch identifier.
        step: Global step count.
        l_dict: Dictionary of loss components.
        tb_logger: TensorBoard summary writer.
        opts: Options dictionary.
    """
    avg_cost: float = cost.mean().item()
    norms, norms_clipped = grad_norms

    # Log values to screen
    print(
        "{}: {}, train_batch_id: {}, avg_cost: {}".format(
            "day" if opts["train_time"] else "epoch", epoch, batch_id, avg_cost
        )
    )
    print("grad_norm: {}, clipped: {}".format(norms[0], norms_clipped[0]))

    # Log values to tensorboard
    if not opts["no_tensorboard"]:
        tb_logger.add_scalar("avg_cost", avg_cost, step)
        tb_logger.add_scalar("actor_loss", l_dict["reinforce_loss"].mean().item(), step)
        tb_logger.add_scalar("nll", l_dict["nll"].mean().item(), step)
        tb_logger.add_scalar("grad_norm", norms[0].item(), step)
        tb_logger.add_scalar("grad_norm_clipped", norms_clipped[0], step)
        if "imitation_loss" in l_dict:
            tb_logger.add_scalar("imitation_loss", l_dict["imitation_loss"].item(), step)
        if opts["baseline"] == "critic":
            tb_logger.add_scalar("critic_loss", l_dict["baseline_loss"].item(), step)
            tb_logger.add_scalar("critic_grad_norm", norms[1].item(), step)
            tb_logger.add_scalar("critic_grad_norm_clipped", norms_clipped[1].item(), step)

    if opts["wandb_mode"] != "disabled":
        wandb_data = {
            "avg_cost": avg_cost,
            "actor_loss": l_dict["reinforce_loss"].mean().item(),
            "nll": l_dict["nll"].mean().item(),
            "grad_norm": norms[0].item(),
            "grad_norm_clipped": norms_clipped[0],
        }
        if "imitation_loss" in l_dict:
            wandb_data["imitation_loss"] = l_dict["imitation_loss"].item()
        wandb.log(wandb_data)
        if opts["baseline"] == "critic":
            wandb.log(
                {
                    "critic_loss": l_dict["baseline_loss"].item(),
                    "critic_grad_norm": norms[1].item(),
                    "critic_grad_norm_clipped": norms_clipped[1].item(),
                }
            )

    if "imitation_loss" in l_dict and l_dict["imitation_loss"].item() != 0:
        print(f"imitation_loss: {l_dict['imitation_loss'].item():.6f}")

    return


def log_epoch(
    x_tup: Tuple[str, int],
    loss_keys: List[str],
    epoch_loss: Dict[str, List[torch.Tensor]],
    opts: Dict[str, Any],
) -> None:
    """
    Logs summary statistics for a completed epoch.

    Args:
        x_tup: Tuple containing (label, value) for the x-axis (e.g., ("epoch", 1)).
        loss_keys: List of loss keys to log.
        epoch_loss: Dictionary collecting losses for the epoch.
        opts: Options dictionary.
    """
    log_str: str = f"Finished {x_tup[0]} {x_tup[1]} log:"
    for id, key in enumerate(loss_keys):
        if not epoch_loss.get(key):
            continue
        lname: str = key if key in udef.LOSS_KEYS else f"{key}_cost"
        # Handle cases where epoch_loss[key] might be empty or contain non-tensors
        try:
            lmean: float = torch.cat(epoch_loss[key]).float().mean().item()
        except Exception:
            lmean = 0.0

        log_str += f" {lname}: {lmean:.4f}"
        if opts["wandb_mode"] != "disabled":
            wandb.log({x_tup[0]: x_tup[1], lname: lmean}, commit=(key == loss_keys[-1]))
    print(log_str)
    return


def get_loss_stats(epoch_loss: Dict[str, List[torch.Tensor]]) -> List[float]:
    """
    Computes mean, std, min, and max for each loss key in the epoch data.

    Args:
        epoch_loss: Dictionary of loss sets.

    Returns:
        Flattened list of [mean, std, min, max] for each key.
    """
    loss_stats: List[float] = []
    for key in epoch_loss.keys():
        loss_tensor: torch.Tensor = torch.cat(epoch_loss[key]).float()
        loss_tmp: List[float] = [
            torch.mean(loss_tensor).item(),
            torch.std(loss_tensor).item(),
            torch.min(loss_tensor).item(),
            torch.max(loss_tensor).item(),
        ]
        loss_stats.extend(loss_tmp)
    return loss_stats


def _sort_log(log: Dict[str, Any]) -> Dict[str, Any]:
    log = {key: value for key, value in sorted(log.items())}
    tmp_log: Dict[str, Any] = {}
    for key in log.keys():
        if "policy_last_minute" in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if "policy_regular" in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if "policy_look_ahead" in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if "gurobi" in key:
            tmp_log[key] = log[key]
    for key in log.keys():
        if "hexaly" in key:
            tmp_log[key] = log[key]
    for key in tmp_log.keys():
        log[key] = log.pop(key)
    return log


def sort_log(logfile_path: str, lock: Optional[threading.Lock] = None) -> None:
    """
    Sorts a JSON log file by grouping keys (policies, solvers).

    Args:
        logfile_path: Path to the log file.
        lock: Thread lock.
    """
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return
    try:
        log: Dict[str, Any] = cast(Dict[str, Any], read_json(logfile_path, lock=None))
        log = _sort_log(log)
        with open(logfile_path, "w") as fp:
            json.dump(log, fp, indent=True)
    finally:
        if lock is not None:
            lock.release()
    return


def _convert_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


def log_to_json(
    json_path: str,
    keys: List[str],
    dit: Dict[str, Any],
    sort_log: bool = True,
    sample_id: Optional[int] = None,
    lock: Optional[threading.Lock] = None,
) -> Union[Dict[str, Any], List[Any]]:
    """
    Writes data to a JSON file, handling updates and sorting.
    (Legacy version)

    Args:
        json_path: Path to the output JSON file.
        keys: List of keys for the inner dictionary.
        dit: Data to write.
        sort_log: Whether to sort keys. Defaults to True.
        sample_id: Index for list-based logs.
        lock: Thread lock.

    Returns:
        The updated log data.
    """
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return [] if sample_id is not None else {}
    try:
        read_failed: bool = False
        old: Union[Dict[str, Any], List[Any]]
        if os.path.isfile(json_path):
            try:
                old = read_json(json_path, lock=None)
            except json.JSONDecodeError:
                read_failed = True
                old = [] if "full" in json_path else {}

            if sample_id is not None and isinstance(old, list) and len(old) > sample_id:
                new: Dict[str, Any] = old[sample_id]
            elif isinstance(old, dict):
                new = old
            else:
                new = {}
        else:
            new = {}
            old = [] if sample_id is not None else {}

        for key, val in dit.items():
            values: Iterable[Any] = val.values() if isinstance(val, dict) else val
            new[key] = dict(zip(keys, values))

        if sort_log:
            new = _sort_log(new)
        if sample_id is not None:
            if isinstance(old, list):
                if len(old) > sample_id:
                    old[sample_id] = new
                else:
                    old.append(new)
            elif isinstance(old, dict):
                old[str(sample_id)] = new
        else:
            old = new

        write_error: Optional[Exception] = None
        try:
            with open(json_path, "w") as fp:
                json.dump(_convert_numpy(old), fp, indent=True)
        except (TypeError, ValueError, FileNotFoundError) as e:
            write_error = e

        if read_failed or write_error is not None:
            timestamp: str = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            filename, file_ext = os.path.splitext(json_path)
            tmp_path: str = filename + timestamp + "_TMP" + file_ext
            with open(tmp_path, "w") as fp_temp:
                json.dump(_convert_numpy(old), fp_temp, indent=True)

            if read_failed and write_error is not None:
                print(f"\n[WARNING] Failed to read from and write to {json_path}.\n- Write Error: {write_error}")
            elif read_failed:
                print(f"\n[WARNING] Failed to read from {json_path}.")
            else:
                print(f"\n[WARNING] Failed to write to {json_path}.\n- Write Error: {write_error}")

            print(f"Data saved to temporary file: {tmp_path}")
    finally:
        if lock is not None:
            lock.release()
    return old


def log_to_json2(
    json_path: str,
    keys: List[str],
    dit: Dict[str, Any],
    sort_log: bool = True,
    sample_id: Optional[int] = None,
    lock: Optional[threading.Lock] = None,
) -> Union[Dict[str, Any], List[Any]]:
    """
    Thread-safe and error-resilient JSON logger.
    Handles read/write errors gracefully with temporary file fallbacks.

    Args:
        json_path: Path to the output JSON file.
        keys: List of keys used if data items are sequences.
        dit: Dictionary of data to log.
        sort_log: Whether to sort the log keys. Defaults to True.
        sample_id: ID to update a specific list entry.
        lock: Thread lock.

    Returns:
        The final data structure written (or attempted to write).
    """
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return [] if sample_id is not None else {}

    data_to_write: Union[Dict[str, Any], List[Any]]
    try:
        old: Union[Dict[str, Any], List[Any]]
        if os.path.isfile(json_path):
            try:
                old = read_json(json_path, lock=None)
            except json.JSONDecodeError:
                print(f"\n[WARNING] JSONDecodeError reading: {json_path}. Assuming empty data for merge.")
                old = [] if "full" in json_path else {}

            if sample_id is not None and isinstance(old, list) and len(old) > sample_id:
                new: Dict[str, Any] = old[sample_id]
            elif isinstance(old, dict):
                new = old
            else:
                new = {}
        else:
            new = {}
            old = [] if sample_id is not None else {}

        for key, val in dit.items():
            values: Iterable[Any] = val.values() if isinstance(val, dict) else val
            new[key] = dict(zip(keys, values))

        if sort_log:
            new = _sort_log(new)

        if sample_id is not None:
            if isinstance(old, list):
                if len(old) > sample_id:
                    old[sample_id] = new
                else:
                    old.append(new)
            elif isinstance(old, dict):
                old[str(sample_id)] = new
        else:
            old = new

        data_to_write = old

        try:
            with open(json_path, "w") as fp:
                json.dump(_convert_numpy(data_to_write), fp, indent=True)

        except (TypeError, ValueError) as e:
            timestamp: str = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            temp_json_path: str = json_path + timestamp + "_TEMP.json"

            with open(temp_json_path, "w") as fp_temp:
                json.dump(_convert_numpy(data_to_write), fp_temp, indent=True)

            print(f"\n[ERROR] Failed to write to {json_path}. Data saved to temporary file: {temp_json_path}")
            print(f"Original Write Error: {e}")

    finally:
        if lock is not None:
            lock.release()

    return data_to_write


def log_to_pickle(
    pickle_path: str,
    log: Any,
    lock: Optional[threading.Lock] = None,
    dw_func: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Logs data to a pickle file.

    Args:
        pickle_path: Output path.
        log: Data to pickle.
        lock: Thread lock.
        dw_func: Post-write callback function.
    """
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return
    try:
        with open(pickle_path, "wb") as file:
            pickle.dump(log, file)
        if dw_func is not None:
            dw_func(pickle_path)
    finally:
        if lock is not None:
            lock.release()
    return


def update_log(
    json_path: str,
    new_output: List[Dict[str, Any]],
    start_id: int,
    policies: List[str],
    sort_log: bool = True,
    lock: Optional[threading.Lock] = None,
) -> Union[Dict[str, Any], List[Any]]:
    """
    Updates specific entries in a JSON log file.

    Args:
        json_path: Path to the log file.
        new_output: New data to merge in.
        start_id: Starting index for the merge.
        policies: List of policies to update.
        sort_log: Whether to sort the result. Defaults to True.
        lock: Thread lock.

    Returns:
        The updated log data.
    """
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return {}
    try:
        new_logs: Union[Dict[str, Any], List[Any]]
        try:
            new_logs = read_json(json_path, lock=None)
        except json.JSONDecodeError:
            new_logs = [] if "full" in json_path else {}

        for id, log in enumerate(new_output):
            if isinstance(new_logs, list):
                target = new_logs[start_id + id]
            else:
                target = new_logs[str(start_id + id)]

            for pol in policies:
                target[pol] = log[pol]

            if sort_log:
                if isinstance(new_logs, list):
                    new_logs[start_id + id] = _sort_log(target)
                else:
                    new_logs[str(start_id + id)] = _sort_log(target)

        with open(json_path, "w") as fp:
            json.dump(new_logs, fp, indent=True)
    finally:
        if lock is not None:
            lock.release()
    return new_logs


@compose_dirpath
def load_log_dict(
    dir_paths: List[str],
    nsamples: List[int],
    show_incomplete: bool = False,
    lock: Optional[threading.Lock] = None,
) -> Dict[str, str]:
    """
    Loads log filenames for multiple graph sizes/runs.

    Args:
        dir_paths: List of directory paths.
        nsamples: Number of samples corresponding to each dir.
        show_incomplete: Check for incomplete runs. Defaults to False.
        lock: Thread lock.

    Returns:
        Mapping of graph size to log file path.
    """
    assert len(dir_paths) == len(
        nsamples
    ), f"Len of dir_paths and nsamples lists must be equal, not {len(dir_paths)} != {len(nsamples)}"
    logs: Dict[str, str] = {}
    for path, ns in zip(dir_paths, nsamples):
        gsize: int = int(os.path.basename(path).split("_")[1])
        logs[f"{gsize}"] = os.path.join(path, f"log_mean_{ns}N.json")
        if show_incomplete and ns > 1:
            counter: Counter[str] = Counter()
            log_full: List[Dict[str, Any]] = cast(
                List[Dict[str, Any]], read_json(os.path.join(path, f"log_full_{ns}N.json"), lock)
            )
            for run in log_full:
                counter.update(run.keys())

            for key, val in dict(counter).items():
                if ns - val > 0:
                    print(f"graph {gsize} incomplete runs:")
                    print("-", key, "-", ns - val)
    return logs

    return


@compose_dirpath
def output_stats(
    dir_path: str,
    nsamples: int,
    policies: List[str],
    keys: List[str],
    sort_log: bool = True,
    print_output: bool = False,
    lock: Optional[threading.Lock] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calculates and saves mean and standard deviation of log data.

    Args:
        dir_path: Directory containing logs.
        nsamples: Number of samples.
        policies: Policies to analyze.
        keys: Keys to extract stats for.
        sort_log: Sort the output. Defaults to True.
        print_output: Print stats to console. Defaults to False.
        lock: Thread lock.

    Returns:
        tuple: (mean_dict, std_dict)
    """
    mean_filename: str = os.path.join(dir_path, f"log_mean_{nsamples}N.json")
    std_filename: str = os.path.join(dir_path, f"log_std_{nsamples}N.json")
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return {}, {}
    try:
        mean_dit: Dict[str, Any]
        std_dit: Dict[str, Any]
        if os.path.isfile(mean_filename):
            mean_dit = cast(Dict[str, Any], read_json(mean_filename, lock=None))
            std_dit = cast(Dict[str, Any], read_json(std_filename, lock=None))
        else:
            mean_dit = {}
            std_dit = {}

        log: str = os.path.join(dir_path, f"log_full_{nsamples}N.json")
        data: List[Dict[str, Any]] = cast(List[Dict[str, Any]], read_json(log, lock=None))
        for pol in policies:
            tmp: List[Sequence[float]] = []
            for n_id in range(nsamples):
                tmp.append(list(data[n_id][pol].values()))
            mean_dit[pol] = {key: val for key, val in zip(keys, [*map(statistics.mean, zip(*tmp))])}
            std_dit[pol] = {key: val for key, val in zip(keys, [*map(statistics.stdev, zip(*tmp))])}

        if sort_log:
            mean_dit = _sort_log(mean_dit)
            std_dit = _sort_log(std_dit)
        if print_output:
            for pol in mean_dit.keys():
                lg = mean_dit[pol]
                lg_std = std_dit[pol]
                logm: Iterable[float] = lg.values() if isinstance(lg, dict) else lg
                logs: Iterable[float] = lg_std.values() if isinstance(lg_std, dict) else lg_std
                tmp_lg: List[Tuple[str, str]] = [(str(x), str(y)) for x, y in zip(logm, logs)]
                if pol in policies:
                    print(f"{pol}:")
                    for (x, y), key in zip(tmp_lg, keys):
                        print(f"- {key} value: {x[: x.find('.') + 3]} +- {y[: y.find('.') + 5]}")

        with open(mean_filename, "w") as fp:
            json.dump(mean_dit, fp, indent=True)
        with open(std_filename, "w") as fp:
            json.dump(std_dit, fp, indent=True)
    finally:
        if lock is not None:
            lock.release()
    return mean_dit, std_dit


@compose_dirpath
def runs_per_policy(
    dir_paths: List[str],
    nsamples: List[int],
    policies: List[str],
    print_output: bool = False,
    lock: Optional[threading.Lock] = None,
) -> List[Dict[str, List[int]]]:
    """
    Counts successful runs for each policy in given directories.

    Args:
        dir_paths: List of directories.
        nsamples: List of expected sample counts.
        policies: List of policies.
        print_output: Print results. Defaults to False.
        lock: Thread lock.

    Returns:
        List of dictionaries of run IDs per policy.
    """
    assert len(dir_paths) == len(
        nsamples
    ), f"Len of dir_paths and nsamples lists must be equal, not {len(dir_paths)} != {len(nsamples)}"
    runs_ls: List[Dict[str, List[int]]] = []
    for path, ns in zip(dir_paths, nsamples):
        dit: Dict[str, List[int]] = {pol: [] for pol in policies}
        log: str = os.path.join(path, f"log_full_{ns}N.json")
        data: List[Dict[str, Any]] = cast(List[Dict[str, Any]], read_json(log, lock))
        for id, run_data in enumerate(data):
            for key in dit:
                if key in run_data:
                    dit[key].append(id)

        runs_ls.append(dit)
        if print_output:
            gsize: int = int(os.path.basename(path).rsplit("_", 1)[1])
            print(f"graph {gsize} #runs per policy:")
            for key, val in dit.items():
                print(f"- {key}: {len(val)}")
                print(" -- sample IDs:", val)
    return runs_ls


def send_daily_output_to_gui(
    daily_log: Dict[str, Any],
    policy: str,
    sample_idx: int,
    day: int,
    bins_c: Sequence[float],
    collected: Sequence[float],
    bins_c_after: Sequence[float],
    log_path: str,
    tour: Sequence[int],
    coordinates: Union[pd.DataFrame, List[Any]],
    lock: Optional[threading.Lock] = None,
) -> None:
    """
    Formats and writes daily simulation stats to a log file for the GUI.

    Args:
        daily_log: Metrics for the day.
        policy: Policy name.
        sample_idx: Sample ID.
        day: Day index.
        bins_c: Bin levels before collection.
        collected: Amount collected.
        bins_c_after: Bin levels after collection.
        log_path: Output log file path.
        tour: Route indices.
        coordinates: Node coordinates.
        lock: Thread lock.
    """
    full_payload: Dict[str, Any] = {k: v for k, v in daily_log.items() if k in udef.DAY_METRICS[:-1]}
    route_coords: List[Dict[str, Any]] = []

    # Prepare coordinates lookup: normalize headers and handle potential duplicates
    coords_lookup: Optional[pd.DataFrame] = None
    if isinstance(coordinates, pd.DataFrame):
        coords_lookup = coordinates.copy()
        # Normalize headers to UPPERCASE and STRIP whitespace
        coords_lookup.columns = [str(c).upper().strip() for c in coords_lookup.columns]

    for idx in tour:
        # Handle Depot (0) or Bins
        point_data: Dict[str, Any] = {"id": str(idx), "type": "bin"}

        if idx == 0:
            point_data["type"] = "depot"
            point_data["popup"] = "Depot"

        # Safe extraction of coordinates
        if coords_lookup is not None:
            # Check for ID in index (handling int vs str mismatch)
            lookup_idx: Union[int, str] = idx
            if lookup_idx not in coords_lookup.index:
                if str(idx) in coords_lookup.index:
                    lookup_idx = str(idx)
                elif isinstance(idx, (int, np.integer)) and int(idx) in coords_lookup.index:
                    lookup_idx = int(idx)

            if lookup_idx in coords_lookup.index:
                try:
                    row = coords_lookup.loc[lookup_idx]
                    # If duplicate indices exist, take the first one
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]

                    # Look for standard Latitude/Longitude columns
                    lat_val = row.get("LAT", row.get("LATITUDE"))
                    lng_val = row.get("LNG", row.get("LONGITUDE", row.get("LONG")))

                    if lat_val is not None and lng_val is not None:
                        # Parse, replacing comma if locale issue
                        lat: float = float(str(lat_val).replace(",", "."))
                        lng: float = float(str(lng_val).replace(",", "."))
                        point_data["lat"] = lat
                        point_data["lng"] = lng

                        if idx != 0:
                            # Try to get a meaningful ID for popup, fallback to index
                            display_id: Any = row.get("ID", idx)
                            point_data["popup"] = f"ID {display_id}"
                except Exception:
                    pass

        route_coords.append(point_data)

    full_payload.update({udef.DAY_METRICS[-1]: route_coords})
    full_payload.update(
        {
            "bin_state_c": list(bins_c),
            "bin_state_collected": list(collected),
            "bins_state_c_after": list(bins_c_after),
        }
    )
    log_msg: str = f"GUI_DAY_LOG_START:{policy},{sample_idx},{day},{json.dumps(full_payload)}"

    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return
    try:
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")
            f.flush()
    except Exception as e:
        print(f"Warning: Failed to write to local log file: {e}")
    finally:
        if lock is not None:
            lock.release()
    return


def send_final_output_to_gui(
    log: Dict[str, Any],
    log_std: Optional[Dict[str, Any]],
    n_samples: int,
    policies: List[str],
    log_path: str,
    lock: Optional[threading.Lock] = None,
) -> None:
    """
    Formats and writes final simulation summary to a log file for the GUI.

    Args:
        log: Mean stats.
        log_std: Standard deviation stats.
        n_samples: Number of samples.
        policies: List of policies.
        log_path: Output log file path.
        lock: Thread lock.
    """
    lgsd: Dict[str, Any] = (
        {k: [[0] * len(v) if isinstance(v, (tuple, list)) else 0 for v in pol_data] for k, pol_data in log.items()}
        if log_std is None
        else {k: [list(v) if isinstance(v, (tuple, list)) else v for v in pol_data] for k, pol_data in log_std.items()}
    )
    summary_data: Dict[str, Any] = {
        "log": {k: [list(v) if isinstance(v, (tuple, list)) else v for v in pol_data] for k, pol_data in log.items()},
        "log_std": lgsd,
        "n_samples": n_samples,
        "policies": policies,
    }

    summary_message: str = f"GUI_SUMMARY_LOG_START: {json.dumps(summary_data)}"

    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return
    try:
        with open(log_path, "a") as f:
            f.write(summary_message + "\n")
            f.flush()
    except Exception as e:
        print(f"Warning: Failed to write summary to local log file: {e}")
    finally:
        if lock is not None:
            lock.release()
    return


def final_simulation_summary(log: Dict[str, Any], policy: str, n_samples: int) -> None:
    """
    Logs a high-level summary of the simulation results.
    """
    if policy not in log:
        logger.warning(f"Policy {policy} not found in log for summary.")
        return

    stats: Dict[str, float] = log[policy]
    logger.info(f"=== Simulation Summary: {policy} ({n_samples} samples) ===")
    for metric in ["overflows", "kg", "km", "kg/km", "profit"]:
        if metric in stats:
            val: float = stats[metric]  # type: ignore
            logger.info(f" - {metric:10}: {val:>10.2f}")
    logger.info("================================================")
