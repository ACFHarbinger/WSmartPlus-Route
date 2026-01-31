"""
Persistent storage utilities (JSON, Pickle) and system logging.
"""
import datetime
import json
import os
import pickle
import threading
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
from loguru import logger

import logic.src.constants as udef
from logic.src.utils.io.files import read_json


def setup_system_logger(log_path: str = "logs/system.log", level: str = "INFO") -> Any:
    """Configures loguru to log to both console and a file."""
    import sys

    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(log_path, rotation="10 MB", level=level)
    return logger


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


def _sort_log(log: Dict[str, Any]) -> Dict[str, Any]:
    log = {key: value for key, value in sorted(log.items())}
    tmp_log: Dict[str, Any] = {}
    keywords = ["policy_last_minute", "policy_regular", "policy_look_ahead", "gurobi", "hexaly"]
    for kw in keywords:
        for key in log.keys():
            if kw in key:
                tmp_log[key] = log[key]
    for key in tmp_log.keys():
        log[key] = log.pop(key)
    return log


def sort_log(logfile_path: str, lock: Optional[threading.Lock] = None) -> None:
    """Sorts a JSON log file by grouping keys."""
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
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return [] if sample_id is not None else {}
    try:
        if os.path.isfile(json_path):
            try:
                old = read_json(json_path, lock=None)
            except json.JSONDecodeError:
                old = [] if "full" in json_path else {}
            if sample_id is not None and isinstance(old, list) and len(old) > sample_id:
                new = old[sample_id]
            elif isinstance(old, dict):
                new = old
            else:
                new = {}
        else:
            new = {}
            old = [] if sample_id is not None else {}

        for key, val in dit.items():
            values = val.values() if isinstance(val, dict) else val
            new[key] = dict(zip(keys, values))

        if sort_log_flag:
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

        try:
            with open(json_path, "w") as fp:
                json.dump(_convert_numpy(old), fp, indent=True)
        except Exception as e:
            timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            filename, file_ext = os.path.splitext(json_path)
            tmp_path = filename + timestamp + "_TMP" + file_ext
            with open(tmp_path, "w") as fp_temp:
                json.dump(_convert_numpy(old), fp_temp, indent=True)
            print(f"[WARNING] Failed to write to {json_path}. Saved to {tmp_path}. Error: {e}")
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
    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return [] if sample_id is not None else {}
    try:
        if os.path.isfile(json_path):
            try:
                old = read_json(json_path, lock=None)
            except json.JSONDecodeError:
                old = [] if "full" in json_path else {}
            if sample_id is not None and isinstance(old, list) and len(old) > sample_id:
                new = old[sample_id]
            elif isinstance(old, dict):
                new = old
            else:
                new = {}
        else:
            new = {}
            old = [] if sample_id is not None else {}
        for key, val in dit.items():
            values = val.values() if isinstance(val, dict) else val
            new[key] = dict(zip(keys, values))
        if sort_log_flag:
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
        try:
            with open(json_path, "w") as fp:
                json.dump(_convert_numpy(old), fp, indent=True)
        except Exception as e:
            timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
            temp_path = json_path + timestamp + "_TEMP.json"
            with open(temp_path, "w") as fp_temp:
                json.dump(_convert_numpy(old), fp_temp, indent=True)
            print(f"[ERROR] Failed to write to {json_path}. Saved to {temp_path}. Error: {e}")
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
