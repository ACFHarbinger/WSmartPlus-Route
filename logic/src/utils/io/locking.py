"""
Persistence utilities (locking, thread-safe I/O).
"""

import json

import pandas as pd

import logic.src.utils.definitions as udef


def read_output(json_path, policies, lock=None):
    """
    Reads a JSON file and extracts specific policy values into a transposed list.

    Args:
        json_path (str): Path to the JSON file.
        policies (list): List of policy keys to extract.
        lock (threading.Lock, optional): Thread lock for safe access.

    Returns:
        list: Transposed list of values for the specified policies.
    """
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        with open(json_path) as json_file:
            json_data = json.load(json_file)
    finally:
        if lock is not None:
            lock.release()
    tmp = []
    for key, val in json_data.items():
        if key in policies:
            tmp.append(val)
    tmp = pd.DataFrame(tmp).values.transpose().tolist()
    return tmp
