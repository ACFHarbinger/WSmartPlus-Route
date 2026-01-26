"""Tests for log_utils.py."""

import numpy as np
from logic.src.utils.logging.log_utils import _convert_numpy, _sort_log


def test_convert_numpy():
    """Verify NumPy to Python conversion."""
    data = {
        "scalar": np.int64(1),
        "array": np.array([1.0, 2.0]),
        "nested": {"val": np.float32(0.5)},
        "list": [np.int32(10)],
    }
    converted = _convert_numpy(data)
    assert isinstance(converted["scalar"], int)
    assert isinstance(converted["array"], list)
    assert isinstance(converted["nested"]["val"], float)
    assert isinstance(converted["list"][0], int)


def test_sort_log():
    """Verify log dictionary sorting."""
    log = {"z": 1, "policy_regular_test": 10, "a": 5, "gurobi_test": 20}
    sorted_log = _sort_log(log)
    # The keys should be sorted but some (policy_regular, gurobi) are prioritized at the end?
    # Actually _sort_log puts policy_last_minute, regular, look_ahead, gurobi, hexaly at the END
    keys = list(sorted_log.keys())
    # "a" (sorted), "z" (sorted), "gurobi_test" (end), "policy_regular_test" (end)
    # Wait, the logic in _sort_log:
    # 1. sort all items alphabetically
    # 2. collect matches for policies in tmp_log
    # 3. pop matches and move to end
    assert keys[0] == "a"
    assert keys[1] == "z"
    # gurobi comes before policy_regular if they were in that order in the sorting loop
    # but the loop matches policies in a specific order: last_minute, regular, look_ahead, gurobi, hexaly
    # and pops them in that order.
    # so gurobi should be after regular?
    # loop 1: last_minute
    # loop 2: regular -> found policy_regular_test
    # loop 3: look_ahead
    # loop 4: gurobi -> found gurobi_test
    # pop loop: regular then gurobi
    assert "policy_regular_test" in keys[-2:]
    assert "gurobi_test" in keys[-2:]
