"""Unit tests for storage.py."""

import os
import json
import pickle
import threading
import torch
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from logic.src.utils.logging.modules.storage import (
    _convert_numpy,
    _sort_log,
    sort_log,
    log_to_json,
    log_to_json2,
    log_to_pickle,
    update_log,
    setup_system_logger
)

def test_convert_numpy():
    """Test conversion of numpy types to standard python types for JSON."""
    data = {
        "arr": np.array([1, 2, 3]),
        "int": np.int64(10),
        "float": np.float32(0.5),
        "nested": {"list": [np.int32(1)]}
    }
    converted = _convert_numpy(data)
    assert isinstance(converted["arr"], list)
    assert isinstance(converted["int"], int)
    assert isinstance(converted["float"], float)
    assert isinstance(converted["nested"]["list"][0], int)

def test_sort_log_dict():
    """Test grouping of log keys."""
    log = {
        "z_other": 1,
        "policy_regular_v1": 10,
        "a_other": 2,
        "gurobi_v1": 20
    }
    sorted_log = _sort_log(log)
    # Keywords are: ["policy_last_minute", "policy_regular", "policy_look_ahead", "gurobi", "hexaly"]
    # Sorted order (alphabetical keys) then keywords moved to end?
    # Wait, the code says:
    # log = {key: value for key, value in sorted(log.items())}  -> a_other, gurobi_v1, policy_regular_v1, z_other
    # keywords loop: gurobi_v1, policy_regular_v1 added to tmp_log
    # second loop: log.pop(key) for keys in tmp_log -> moves them to end in keyword order.
    # Keyword order: policy_regular, then gurobi.
    keys = list(sorted_log.keys())
    assert keys[0] == "a_other"
    assert keys[1] == "z_other"
    assert keys[2] == "policy_regular_v1"
    assert keys[3] == "gurobi_v1"

def test_sort_log_file(tmp_path):
    """Test sorting a log file."""
    log_path = tmp_path / "test.json"
    data = {"b": 2, "policy_regular": 1, "a": 3}
    with open(log_path, "w") as f:
        json.dump(data, f)

    sort_log(str(log_path))

    with open(log_path, "r") as f:
        loaded = json.load(f)
    assert list(loaded.keys()) == ["a", "b", "policy_regular"]

def test_log_to_json_new_file(tmp_path):
    """Test logging to a new JSON file."""
    json_path = str(tmp_path / "log.json")
    keys = ["k1", "k2"]
    dit = {"sample1": {"k1": 10, "k2": 20}}

    log_to_json(json_path, keys, dit, sort_log_flag=False)

    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        loaded = json.load(f)
    assert loaded["sample1"]["k1"] == 10

def test_log_to_json_with_sample_id(tmp_path):
    """Test logging to a list-based JSON file using sample_id."""
    json_path = str(tmp_path / "full_log.json") # "full" triggers list initialization
    keys = ["cost"]

    # First sample
    log_to_json(json_path, keys, {"pol": [10.0]}, sample_id=0)
    # Second sample
    log_to_json(json_path, keys, {"pol": [20.0]}, sample_id=1)

    with open(json_path, "r") as f:
        loaded = json.load(f)
    assert isinstance(loaded, list)
    assert len(loaded) == 2
    assert loaded[0]["pol"]["cost"] == 10.0
    assert loaded[1]["pol"]["cost"] == 20.0

def test_log_to_pickle(tmp_path):
    """Test pickle logging."""
    path = str(tmp_path / "test.pkl")
    data = {"hello": "world"}
    log_to_pickle(path, data)

    assert os.path.exists(path)
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == data

def test_update_log(tmp_path):
    """Test updating existing log entries."""
    json_path = str(tmp_path / "update.json")
    # Initial state: list of 2 dicts
    initial = [{"p1": {"c": 1}}, {"p1": {"c": 2}}]
    with open(json_path, "w") as f:
        json.dump(initial, f)

    new_output = [{"p2": {"v": 10}}, {"p2": {"v": 20}}]
    update_log(json_path, new_output, start_id=0, policies=["p2"])

    with open(json_path, "r") as f:
        loaded = json.load(f)
    assert loaded[0]["p1"]["c"] == 1
    assert loaded[0]["p2"]["v"] == 10

@patch("logic.src.utils.logging.modules.storage.logger")
def test_setup_system_logger(mock_logger):
    """Test system logger configuration."""
    setup_system_logger("test.log")
    assert mock_logger.remove.called
    assert mock_logger.add.called

def test_log_to_json_acquired_lock_failure():
    """Test behavior when lock cannot be acquired."""
    lock = MagicMock()
    lock.acquire.return_value = False
    res = log_to_json("path", [], {}, lock=lock)
    assert res == {}

    res_sample = log_to_json("path", [], {}, sample_id=0, lock=lock)
    assert res_sample == []
