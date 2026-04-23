"""Persistent storage and system logging utilities.

This module provides tools for serializing experiment data to disk using JSON
and Pickle formats. It includes specialized converters for NumPy types,
thread-safe file operations with adaptive locking, and configuration for
the system-wide Loguru logger.

Attributes:
    setup_system_logger: Configures Loguru sink and multi-module log levels.
    sort_log: Reorders a log file by solver policy grouping.
    log_to_json: Persists experiment result dictionaries to a JSON file.
    log_to_json2: Alternative JSON logger with distinct recovery logic.
    log_to_pickle: Serializes generic objects to binary pickle files.
    update_log: Appends new results to an existing structured log.

Example:
    >>> from logic.src.tracking.logging.modules import storage
    >>> storage.log_to_json("results.json", ["km", "kg"], {"greedy": [10, 5]})
"""

import datetime
import json
import logging
import os
import pickle
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
from loguru import logger

import logic.src.constants as udef
from logic.src.utils.io.files import read_json


def setup_system_logger(log_path: str = "logs/system.log", level: str = "INFO") -> Any:
    """Configures the Loguru system logger and filters noisy modules.

    Args:
        log_path: File path for the system log. Defaults to "logs/system.log".
        level: Logging severity level. Defaults to "INFO".

    Returns:
        Any: The configured Loguru logger instance.
    """
    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(log_path, rotation="10 MB", level=level)

    noisy_modules = [
        "AttentionDecoder",
        "NeuralAgent",
        "NeuralPolicy",
        "WCContextEmbedder",
        "ContextEmbedder",
    ]
    for module_name in noisy_modules:
        logging.getLogger(module_name).setLevel(logging.WARNING)

    return logger


def _convert_numpy(obj: Any) -> Any:
    """Recursively converts NumPy types to native Python types for JSON serialization.

    Args:
        obj: The object to convert (scalar, list, or dictionary).

    Returns:
        Any: The converted object with native Python primitives.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict) or hasattr(obj, "items"):
        return {str(k): _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


def _sort_log(log: Dict[str, Any]) -> Dict[str, Any]:
    """Sorts log entries so that specific policy keywords appear last.

    Args:
        log: The dictionary of log results grouped by policy.

    Returns:
        Dict[str, Any]: The sorted dictionary.
    """
    log = {key: value for key, value in sorted(log.items())}
    tmp_log: Dict[str, Any] = {}
    keywords = ["policy_last_minute", "policy_regular", "policy_look_ahead", "gurobi"]
    for kw in keywords:
        for key in log:
            if kw in key:
                tmp_log[key] = log[key]
    for key in tmp_log:
        log[key] = log.pop(key)
    return log


def sort_log(logfile_path: str, lock: Optional[threading.Lock] = None) -> None:
    """Sorts a JSON log file by grouping keys.

    Args:
        logfile_path: Path to the JSON file to sort in-place.
        lock: Optional thread lock for safe file modification. Defaults to None.
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


def log_to_json(
    json_path: str,
    keys: List[str],
    dit: Dict[str, Any],
    sort_log_flag: bool = True,
    sample_id: Optional[int] = None,
    lock: Optional[threading.Lock] = None,
) -> Union[Dict[str, Any], List[Any]]:
    """Write or update a JSON log file with new policy results.

    Args:
        json_path: Path to the JSON log file.
        keys: List of metric names to use as dictionary keys.
        dit: The data dictionary to serialize.
        sort_log_flag: Whether to sort policies after writing. Defaults to True.
        sample_id: Index of the simulation sample (enables list-mode). Defaults to None.
        lock: Optional thread lock for atomic access. Defaults to None.

    Returns:
        Union[Dict[str, Any], List[Any]]: The updated full log state.
    """
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return [] if sample_id is not None else {}
    try:
        if os.path.isfile(json_path):
            try:
                old = read_json(json_path, lock=None)
            except (json.JSONDecodeError, ValueError):
                logger.error(f"Failed to decode {json_path}. Starting fresh.")
                old = [] if sample_id is not None else {}
            if sample_id is not None and isinstance(old, list) and len(old) > sample_id:
                new = cast(Dict[str, Any], old[sample_id])
            elif not isinstance(old, list) and (isinstance(old, dict) or hasattr(old, "get")):
                new = cast(Dict[str, Any], old)
            else:
                new = {}
        else:
            new = {}
            old = [] if sample_id is not None else {}

        for key, val in dit.items():
            values = val.values() if hasattr(val, "values") else val
            new[key] = dict(zip(keys, values, strict=False))

        if sort_log_flag:
            new = _sort_log(new)
        if sample_id is not None:
            if isinstance(old, list):
                if len(old) > sample_id:
                    old[sample_id] = new
                else:
                    old.append(new)
            elif isinstance(old, dict) or hasattr(old, "get"):
                old[str(sample_id)] = new
        else:
            old = new

        try:
            with open(json_path, "w") as fp:
                json.dump(_convert_numpy(old), fp, indent=True)
        except Exception as e:
            timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            filename, file_ext = os.path.splitext(json_path)
            tmp_path = filename + timestamp + "_TMP" + file_ext
            with open(tmp_path, "w") as fp_temp:
                json.dump(_convert_numpy(old), fp_temp, indent=True)
            print(f"\n[WARNING] Failed to write to {json_path}. Saved to {tmp_path}. Error: {e}")
    finally:
        if lock is not None:
            lock.release()
    return old


def log_to_json2(
    json_path: str,
    keys: List[str],
    dit: Dict[str, Any],
    sort_log_flag: bool = True,
    sample_id: Optional[int] = None,
    lock: Optional[threading.Lock] = None,
) -> Union[Dict[str, Any], List[Any]]:
    """Write or update a JSON log file (variant with different recovery logic).

    Args:
        json_path: Path to the JSON log file.
        keys: List of metric names to use as dictionary keys.
        dit: The data dictionary to serialize.
        sort_log_flag: Whether to sort policies after writing. Defaults to True.
        sample_id: Index of the simulation sample (enables list-mode). Defaults to None.
        lock: Optional thread lock for atomic access. Defaults to None.

    Returns:
        Union[Dict[str, Any], List[Any]]: The updated full log state.
    """
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return [] if sample_id is not None else {}
    try:
        if os.path.isfile(json_path):
            try:
                old = read_json(json_path, lock=None)
            except (json.JSONDecodeError, ValueError):
                logger.error(f"Failed to decode {json_path} in log_to_json2.")
                old = [] if sample_id is not None else {}
            if sample_id is not None and isinstance(old, list) and len(old) > sample_id:
                new = cast(Dict[str, Any], old[sample_id])
            elif not isinstance(old, list) and (isinstance(old, dict) or hasattr(old, "get")):
                new = cast(Dict[str, Any], old)
            else:
                new = {}
        else:
            new = {}
            old = [] if sample_id is not None else {}
        for key, val in dit.items():
            values = val.values() if hasattr(val, "values") else val
            new[key] = dict(zip(keys, values, strict=False))
        if sort_log_flag:
            new = _sort_log(new)
        if sample_id is not None:
            if isinstance(old, list):
                if len(old) > sample_id:
                    old[sample_id] = new
                else:
                    old.append(new)
            elif isinstance(old, dict) or hasattr(old, "get"):
                old[str(sample_id)] = new
        else:
            old = new
        try:
            with open(json_path, "w") as fp:
                json.dump(_convert_numpy(old), fp, indent=True)
        except Exception as e:
            timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            temp_path = json_path + timestamp + "_TEMP.json"
            with open(temp_path, "w") as fp_temp:
                json.dump(_convert_numpy(old), fp_temp, indent=True)
            print(f"\n[ERROR] Failed to write to {json_path}. Saved to {temp_path}. Error: {e}")
    finally:
        if lock is not None:
            lock.release()
    return old


def log_to_pickle(
    pickle_path: str,
    log: Any,
    lock: Optional[threading.Lock] = None,
    dw_func: Optional[Callable[[str], None]] = None,
) -> None:
    """Serialize a log object to a pickle file.

    Args:
        pickle_path: Target path for the .pkl file.
        log: The object to serialize.
        lock: Optional thread lock for safe writing. Defaults to None.
        dw_func: Optional callback function triggered after successful write.
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


def update_log(
    json_path: str,
    new_output: List[Dict[str, Any]],
    start_id: int,
    policies: List[str],
    sort_log_flag: bool = True,
    lock: Optional[threading.Lock] = None,
) -> Union[Dict[str, Any], List[Any]]:
    """Update existing log entries with new policy outputs.

    Args:
        json_path: Path to the JSON log file to update.
        new_output: List of new simulation results.
        start_id: The starting sample ID to begin updating from.
        policies: List of policy names to update.
        sort_log_flag: Whether to re-sort results after update. Defaults to True.
        lock: Optional thread lock for atomic updates. Defaults to None.

    Returns:
        Union[Dict[str, Any], List[Any]]: The updated log state.
    """
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return {}
    try:
        try:
            new_logs = read_json(json_path, lock=None)
        except json.JSONDecodeError:
            new_logs = [] if "full" in json_path else {}
        for id, log in enumerate(new_output):
            target = new_logs[start_id + id] if isinstance(new_logs, list) else new_logs[str(start_id + id)]
            for pol in policies:
                target[pol] = log[pol]
            if sort_log_flag:
                target = _sort_log(target)
                if isinstance(new_logs, list):
                    new_logs[start_id + id] = target
                else:
                    new_logs[str(start_id + id)] = target
        with open(json_path, "w") as fp:
            json.dump(new_logs, fp, indent=True)
    finally:
        if lock is not None:
            lock.release()
    return new_logs
