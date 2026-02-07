"""Unit tests for processing.py."""

import os
import json
import pytest
from logic.src.utils.io import (
    process_dict_of_dicts,
    process_list_of_dicts,
    process_dict_two_inputs,
    process_list_two_inputs,
    find_single_input_values,
    find_two_input_values,
    process_pattern_files,
    process_file,
    process_pattern_files_statistics,
    process_file_statistics
)

def test_process_dict_of_dicts():
    """Test updating values in a dict of dicts."""
    data = {
        "a": {"km": 10},
        "b": {"km": [1, 2]},
        "c": {"other": 5}
    }
    def add_val(x, y): return x + y

    modified = process_dict_of_dicts(data, output_key="km", process_func=add_val, update_val=5)
    assert modified is True
    assert data["a"]["km"] == 15
    assert data["b"]["km"] == [6, 7]
    assert data["c"]["other"] == 5

def test_process_list_of_dicts():
    """Test updating values in a list of dicts of dicts."""
    data = [
        {"a": {"km": 10}},
        {"b": {"km": 20}}
    ]
    modified = process_list_of_dicts(data, output_key="km", process_func=lambda x, y: x * y, update_val=2)
    assert modified is True
    assert data[0]["a"]["km"] == 20
    assert data[1]["b"]["km"] == 40

def test_process_dict_two_inputs():
    """Test multi-input calculations in dicts."""
    data = {
        "s1": {"km": 100, "time": 10},
        "s2": {"km": [20, 40], "time": [2, 4]}
    }
    # Speed = km / time
    div = lambda x, y: x / y
    div.__name__ = "/"

    modified = process_dict_two_inputs(data, "km", "time", "speed", div)
    assert modified is True
    assert data["s1"]["speed"] == 10
    assert data["s2"]["speed"] == [10, 10]

def test_find_single_input_values():
    """Test recursive retrieval of values."""
    data = {
        "res": {"km": 10},
        "nested": [{"km": 20}, {"km": 30}]
    }
    vals = find_single_input_values(data, output_key="km")
    # Paths: res, nested[0], nested[1]
    assert len(vals) == 3
    paths = [v[0] for v in vals]
    assert "res" in paths
    assert "nested[0]" in paths

def test_find_two_input_values():
    """Test recursive retrieval of value pairs."""
    data = {
        "a": {"v1": 1, "v2": 2},
        "b": {"v1": [3, 4], "v2": [10, 20]}
    }
    vals = find_two_input_values(data, input_key1="v1", input_key2="v2")
    # pairs: (1, 2), (3, 10), (4, 20)
    assert len(vals) == 3
    assert (1, 2) in [(v[1], v[2]) for v in vals]

def test_process_file(tmp_path):
    """Test processing a single JSON file."""
    path = str(tmp_path / "test.json")
    data = {"pol": {"km": 10}}
    with open(path, "w") as f:
        json.dump(data, f)

    res = process_file(path, output_key="km", process_func=lambda x, y: x + y, update_val=5)
    assert res is True
    with open(path, "r") as f:
        loaded = json.load(f)
    assert loaded["pol"]["km"] == 15

def test_process_file_statistics(tmp_path):
    """Test statistical aggregation from a file."""
    path = str(tmp_path / "input.json")
    # Both paths end with ".v"
    data = {"group1": {"v": 10}, "group2": {"group1": {"v": 20}}}
    with open(path, "w") as f:
        json.dump(data, f)

    out_name = "stats.json"
    res = process_file_statistics(path, output_filename=out_name, output_key="v", process_func=sum)
    assert res is True

    stats_path = str(tmp_path / out_name)
    assert os.path.exists(stats_path)
    with open(stats_path, "r") as f:
        stats = json.load(f)
    assert stats["group1"]["v"] == 30

def test_process_pattern_files(tmp_path):
    """Test processing multiple files matching a pattern."""
    d1 = tmp_path / "sub1"
    d1.mkdir()
    f1 = d1 / "log_1.json"
    with open(f1, "w") as f:
        json.dump({"a": {"km": 1}}, f)

    process_pattern_files(str(tmp_path), filename_pattern="log_*.json", process_func=lambda x, y: x+1)
    with open(f1, "r") as f:
        assert json.load(f)["a"]["km"] == 2
