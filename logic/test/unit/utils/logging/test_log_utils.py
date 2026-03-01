"""Tests for log_utils.py."""

import numpy as np
from logic.src.tracking.logging.modules.storage import _convert_numpy, _sort_log


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
    keys = list(sorted_log.keys())
    assert keys[0] == "a"
    assert keys[1] == "z"

    assert "policy_regular_test" in keys[-2:]
    assert "gurobi_test" in keys[-2:]
