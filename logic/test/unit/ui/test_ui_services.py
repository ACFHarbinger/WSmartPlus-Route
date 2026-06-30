import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Import targets
from logic.src.ui.services.benchmark_loader import (
    get_benchmark_log_path,
    get_unique_benchmarks,
    load_benchmark_data,
)
from logic.src.ui.services.log_parser import (
    DayLogEntry,
    aggregate_metrics_by_day,
    filter_entries,
    get_day_range,
    get_unique_policies,
    get_unique_samples,
    parse_day_log_line,
    parse_log_file,
    stream_log_file,
)
from logic.src.ui.services.simulation_analytics import (
    compute_cumulative_stats,
    compute_day_deltas,
    compute_summary_statistics,
    get_metric_history,
)


# --- Mock Streamlit ---
def mock_cache_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


if "streamlit" not in sys.modules:
    mock_st = MagicMock()
    mock_st.cache_data = mock_cache_decorator
    sys.modules["streamlit"] = mock_st
else:
    mock_st = sys.modules["streamlit"]
    if isinstance(mock_st, MagicMock):
        mock_st.cache_data = mock_cache_decorator

# ==========================================
# 1. Tests for log_parser.py
# ==========================================


def test_parse_day_log_line_valid():
    line = 'GUI_DAY_LOG_START:greedy,0,1,{"profit": 12.5, "overflows": 2}'
    entry = parse_day_log_line(line)
    assert entry is not None
    assert entry.policy == "greedy"
    assert entry.sample_id == 0
    assert entry.day == 1
    assert entry.data == {"profit": 12.5, "overflows": 2}


def test_parse_day_log_line_invalid():
    # Non prefix
    assert parse_day_log_line("SOME_OTHER_LINE") is None
    # Too few parts
    assert parse_day_log_line("GUI_DAY_LOG_START:greedy,0") is None
    # Malformed parts
    assert parse_day_log_line("GUI_DAY_LOG_START:greedy,abc,1,{}") is None
    # Malformed JSON
    assert parse_day_log_line("GUI_DAY_LOG_START:greedy,0,1,{invalid_json}") is None


def test_parse_log_file(tmp_path):
    log_file = tmp_path / "sim.jsonl"
    log_content = (
        'GUI_DAY_LOG_START:greedy,0,1,{"profit": 10, "all_bin_coords": [[1.0, 2.0]]}\n'
        'GUI_DAY_LOG_START:greedy,0,2,{"profit": 15}\n'
    )
    log_file.write_text(log_content)

    entries = parse_log_file(log_file)
    assert len(entries) == 2
    # Verify coordinates backfilled/cached
    assert entries[0].data["all_bin_coords"] == [[1.0, 2.0]]
    assert entries[1].data["all_bin_coords"] == [[1.0, 2.0]]


def test_parse_log_file_non_existent():
    entries = parse_log_file(Path("non_existent_file.jsonl"))
    assert entries == []


def test_stream_log_file(tmp_path):
    log_file = tmp_path / "sim.jsonl"
    log_content = (
        'GUI_DAY_LOG_START:greedy,0,1,{"profit": 10, "all_bin_coords": [[1.0, 2.0]]}\n'
        'GUI_DAY_LOG_START:greedy,0,2,{"profit": 15}\n'
        'GUI_DAY_LOG_START:greedy,0,3,{"profit": 20}\n'
    )
    log_file.write_text(log_content)

    # Stream starting from line index 1
    entries = list(stream_log_file(log_file, start_line=1))
    assert len(entries) == 2
    assert entries[0].day == 2
    # Coordinates should be retrieved from cache even though line 0 was skipped
    assert entries[0].data["all_bin_coords"] == [[1.0, 2.0]]
    assert entries[1].day == 3
    assert entries[1].data["all_bin_coords"] == [[1.0, 2.0]]


def test_stream_log_file_non_existent():
    entries = list(stream_log_file(Path("non_existent_file.jsonl")))
    assert entries == []


def test_get_unique_policies_and_samples():
    entries = [
        DayLogEntry("greedy", 0, 1, {}),
        DayLogEntry("alns", 1, 1, {}),
        DayLogEntry("greedy", 0, 2, {}),
    ]
    assert get_unique_policies(entries) == ["alns", "greedy"]
    assert get_unique_samples(entries) == [0, 1]


def test_filter_entries():
    entries = [
        DayLogEntry("greedy", 0, 1, {}),
        DayLogEntry("alns", 1, 1, {}),
        DayLogEntry("greedy", 0, 2, {}),
    ]
    assert len(filter_entries(entries, policy="greedy")) == 2
    assert len(filter_entries(entries, sample_id=1)) == 1
    assert len(filter_entries(entries, day=1)) == 2
    assert len(filter_entries(entries, policy="greedy", day=2)) == 1


def test_get_day_range():
    assert get_day_range([]) == (0, 0)
    entries = [
        DayLogEntry("greedy", 0, 2, {}),
        DayLogEntry("greedy", 0, 5, {}),
    ]
    assert get_day_range(entries) == (2, 5)


def test_aggregate_metrics_by_day():
    entries = [
        DayLogEntry("greedy", 0, 1, {"profit": 10, "overflows": 2}),
        DayLogEntry("greedy", 1, 1, {"profit": 20, "overflows": 4}),
        DayLogEntry("alns", 0, 1, {"profit": 100}),
    ]
    res = aggregate_metrics_by_day(entries, policy="greedy")
    assert 1 in res
    assert res[1]["profit"] == [10.0, 20.0]
    assert res[1]["overflows"] == [2.0, 4.0]


# ==========================================
# 2. Tests for simulation_analytics.py
# ==========================================


def test_compute_cumulative_stats():
    # Empty
    assert compute_cumulative_stats([]) == {}

    entries = [
        DayLogEntry("greedy", 0, 1, {"profit": 10, "km": 2.0, "kg": 50, "overflows": 1, "cost": 5}),
        DayLogEntry("greedy", 0, 2, {"profit": 20, "km": 3.0, "kg": 100, "overflows": 0, "cost": 10}),
    ]
    totals = compute_cumulative_stats(entries, policy="greedy", sample_id=0)
    assert totals["Total Profit"] == 30.0
    assert totals["Total Distance (km)"] == 5.0
    assert totals["Total Waste (kg)"] == 150.0
    assert totals["Total Overflows"] == 1.0
    assert totals["Total Cost"] == 15.0
    assert totals["Avg Efficiency"] == 30.0  # 150 / 5


def test_compute_day_deltas():
    entries = [
        DayLogEntry("greedy", 0, 1, {"profit": 10}),
        DayLogEntry("greedy", 0, 2, {"profit": 15}),
    ]
    # Current day 2, previous day 1. Delta = 15 - 10 = 5
    deltas = compute_day_deltas(entries, current_day=2, policy="greedy", sample_id=0)
    assert deltas["profit"] == 5.0
    # Other metrics where current/previous is missing should be None
    assert deltas["km"] is None


def test_compute_summary_statistics():
    assert compute_summary_statistics([]) == {}

    entries = [
        DayLogEntry("greedy", 0, 1, {"profit": 10}),
        DayLogEntry("greedy", 0, 2, {"profit": 20}),
    ]
    stats = compute_summary_statistics(entries, policy="greedy")
    assert stats["profit"]["mean"] == 15.0
    assert stats["profit"]["min"] == 10.0
    assert stats["profit"]["max"] == 20.0
    assert stats["profit"]["total"] == 30.0
    # Standard deviation of [10, 20] is 7.071...
    assert stats["profit"]["std"] > 7.0


def test_get_metric_history():
    assert get_metric_history([], "profit") == []

    entries = [
        DayLogEntry("greedy", 0, 1, {"profit": 10}),
        DayLogEntry("greedy", 0, 2, {"profit": 20}),
        DayLogEntry("greedy", 0, 3, {"profit": 30}),
    ]
    # History for last 2 days should be [20.0, 30.0]
    hist = get_metric_history(entries, "profit", policy="greedy", sample_id=0, last_n_days=2)
    assert hist == [20.0, 30.0]


# ==========================================
# 3. Tests for benchmark_loader.py
# ==========================================


def test_get_benchmark_log_path():
    p = get_benchmark_log_path()
    assert isinstance(p, Path)
    assert str(p).endswith("benchmarks.jsonl")


def test_load_benchmark_data_non_existent():
    # Force file path to check a mock nonexistent one
    with patch(
        "logic.src.ui.services.benchmark_loader.get_benchmark_log_path", return_value=Path("non_existent.jsonl")
    ):
        df = load_benchmark_data()
        assert df.empty


def test_load_benchmark_data_exists(tmp_path):
    log_file = tmp_path / "benchmarks.jsonl"
    log_content = (
        '{"type": "performance_benchmark", "timestamp": "2026-06-21T20:00:00Z", "benchmark": "TamModel", "level": "L1", "message": "test", "metrics": {"latency": 5.2}, "metadata": {"device": "cuda"}}\n'
        '{"type": "other_type", "timestamp": "2026-06-21T21:00:00Z"}\n'
        "invalid_json_here\n"
    )
    log_file.write_text(log_content)

    with patch("logic.src.ui.services.benchmark_loader.get_benchmark_log_path", return_value=log_file):
        df = load_benchmark_data()
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["benchmark"] == "TamModel"
        assert df.iloc[0]["latency"] == 5.2
        assert df.iloc[0]["device"] == "cuda"


def test_get_unique_benchmarks():
    df = pd.DataFrame()
    assert get_unique_benchmarks(df) == []

    df = pd.DataFrame([{"benchmark": "B"}, {"benchmark": "A"}, {"benchmark": "B"}])
    assert get_unique_benchmarks(df) == ["A", "B"]
