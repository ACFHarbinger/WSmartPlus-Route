# Copyright (c) WSmart-Route. All rights reserved.
"""
Data loading and caching services for the dashboard.

Provides cached access to training logs and simulation outputs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml

from logic.src.pipeline.ui.services.log_parser import (
    DayLogEntry,
    aggregate_metrics_by_day,
    get_day_range,
    get_unique_policies,
    get_unique_samples,
    parse_log_file,
)

# -----------------------------------------------------------------------------
# Path Configuration
# -----------------------------------------------------------------------------


def get_project_root() -> Path:
    """Get the project root directory (assumes app runs from project root)."""
    return Path.cwd()


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    return get_project_root() / "logs"


def get_simulation_output_dir() -> Path:
    """Get the simulation output directory path."""
    return get_project_root() / "assets" / "output"


# -----------------------------------------------------------------------------
# Training Logs (Lightning)
# -----------------------------------------------------------------------------


@st.cache_data(ttl=60)
def discover_training_runs() -> List[Tuple[str, Path]]:
    """
    Discover all training runs in logs/output.

    Returns:
        List of (version_name, metrics_csv_path) tuples.
    """
    output_dir = get_logs_dir() / "output"
    runs: List[Tuple[str, Path]] = []

    if not output_dir.exists():
        return runs

    for version_dir in sorted(output_dir.glob("version_*")):
        metrics_file = version_dir / "metrics.csv"
        if metrics_file.exists():
            runs.append((version_dir.name, metrics_file))

    return runs


@st.cache_data(ttl=60)
def load_hparams(version_name: str) -> Dict[str, Any]:
    """
    Load hparams.yaml for a training version.

    Args:
        version_name: Version folder name (e.g., "version_0").

    Returns:
        Dict with hyperparameters, or empty dict if not found.
    """
    hparams_path = get_logs_dir() / "output" / version_name / "hparams.yaml"
    if not hparams_path.exists():
        return {}
    try:
        with open(hparams_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_training_metrics(metrics_path: str) -> pd.DataFrame:
    """
    Load training metrics from a Lightning CSV file.

    Args:
        metrics_path: Path to the metrics.csv file.

    Returns:
        DataFrame with training metrics.
    """
    path = Path(metrics_path)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def load_multiple_training_runs(version_names: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load metrics from multiple training runs for comparison.

    Args:
        version_names: List of version folder names (e.g., ["version_0", "version_1"]).

    Returns:
        Dict mapping version name to DataFrame.
    """
    runs = discover_training_runs()
    version_to_path = {name: str(path) for name, path in runs}

    result: Dict[str, pd.DataFrame] = {}
    for version in version_names:
        if version in version_to_path:
            result[version] = load_training_metrics(version_to_path[version])

    return result


# -----------------------------------------------------------------------------
# Simulation Logs
# -----------------------------------------------------------------------------


@st.cache_data(ttl=60)
def discover_simulation_logs() -> List[Tuple[str, Path]]:
    """
    Discover all simulation log files in assets/output.

    Returns:
        List of (relative_path_display, absolute_path) tuples.
    """
    output_dir = get_simulation_output_dir()
    logs: List[Tuple[str, Path]] = []

    if not output_dir.exists():
        return logs

    for jsonl_file in sorted(output_dir.rglob("*.jsonl")):
        # Create a readable display name (relative to output dir)
        rel_path = jsonl_file.relative_to(output_dir)
        logs.append((str(rel_path), jsonl_file))

    return logs


@st.cache_data(ttl=30)
def load_simulation_log(log_path: str) -> List[Dict[str, Any]]:
    """
    Load and parse a simulation log file (cached).

    Args:
        log_path: Absolute path to the .jsonl file.

    Returns:
        List of parsed entry dictionaries.
    """
    path = Path(log_path)
    entries = parse_log_file(path)

    # Convert to dicts for caching (dataclasses may not serialize well)
    return [{"policy": e.policy, "sample_id": e.sample_id, "day": e.day, "data": e.data} for e in entries]


def load_simulation_log_fresh(log_path: str) -> List[DayLogEntry]:
    """
    Load a simulation log without caching (for live tailing).

    Args:
        log_path: Absolute path to the .jsonl file.

    Returns:
        List of DayLogEntry objects.
    """
    return parse_log_file(Path(log_path))


def get_simulation_metadata(entries: List[DayLogEntry]) -> Dict[str, Any]:
    """
    Extract metadata from simulation entries.

    Args:
        entries: List of parsed log entries.

    Returns:
        Dict with policies, samples, day_range.
    """
    return {
        "policies": get_unique_policies(entries),
        "samples": get_unique_samples(entries),
        "day_range": get_day_range(entries),
    }


def entries_to_dataframe(entries: List[DayLogEntry]) -> pd.DataFrame:
    """
    Convert log entries to a pandas DataFrame.

    Args:
        entries: List of DayLogEntry objects.

    Returns:
        DataFrame with flattened metrics.
    """
    rows = []
    for entry in entries:
        row = {
            "policy": entry.policy,
            "sample_id": entry.sample_id,
            "day": entry.day,
        }
        # Flatten numeric metrics from data
        for key in ["overflows", "kg", "km", "kg/km", "cost", "profit", "ncol", "kg_lost"]:
            if key in entry.data:
                row[key] = entry.data[key]
        rows.append(row)

    return pd.DataFrame(rows)


def compute_daily_stats(entries: List[DayLogEntry], policy: Optional[str] = None) -> pd.DataFrame:
    """
    Compute mean and std of metrics per day.

    Args:
        entries: List of log entries.
        policy: Optional policy filter.

    Returns:
        DataFrame with day, metric_mean, metric_std columns.
    """
    aggregated = aggregate_metrics_by_day(entries, policy)

    rows = []
    for day in sorted(aggregated.keys()):
        metrics = aggregated[day]
        row: Dict[str, Any] = {"day": day}
        for metric, values in metrics.items():
            if values:
                import statistics

                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Simulation Analytics
# -----------------------------------------------------------------------------

_METRIC_KEYS = ["profit", "km", "kg", "overflows", "cost", "ncol", "kg_lost", "kg/km"]


def _filter_entries(
    entries: List[DayLogEntry],
    policy: Optional[str] = None,
    sample_id: Optional[int] = None,
) -> List[DayLogEntry]:
    """Filter entries by policy and/or sample_id."""
    result = entries
    if policy:
        result = [e for e in result if e.policy == policy]
    if sample_id is not None:
        result = [e for e in result if e.sample_id == sample_id]
    return result


def compute_cumulative_stats(
    entries: List[DayLogEntry],
    policy: Optional[str] = None,
    sample_id: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute cumulative totals across all days for a policy/sample.

    Args:
        entries: All log entries.
        policy: Optional policy filter.
        sample_id: Optional sample filter.

    Returns:
        Dict with total_profit, total_km, total_kg, total_overflows, total_cost, avg_efficiency.
    """
    filtered = _filter_entries(entries, policy, sample_id)
    if not filtered:
        return {}

    totals: Dict[str, float] = {
        "Total Profit": 0.0,
        "Total Distance (km)": 0.0,
        "Total Waste (kg)": 0.0,
        "Total Overflows": 0.0,
        "Total Cost": 0.0,
    }

    for entry in filtered:
        data = entry.data
        totals["Total Profit"] += data.get("profit", 0)
        totals["Total Distance (km)"] += data.get("km", 0)
        totals["Total Waste (kg)"] += data.get("kg", 0)
        totals["Total Overflows"] += data.get("overflows", 0)
        totals["Total Cost"] += data.get("cost", 0)

    total_km = totals["Total Distance (km)"]
    total_kg = totals["Total Waste (kg)"]
    totals["Avg Efficiency"] = total_kg / total_km if total_km > 0 else 0.0

    return totals


def compute_day_deltas(
    entries: List[DayLogEntry],
    current_day: int,
    policy: Optional[str] = None,
    sample_id: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute metric deltas between current day and previous day.

    Args:
        entries: All log entries.
        current_day: The day to compute deltas for.
        policy: Optional policy filter.
        sample_id: Optional sample filter.

    Returns:
        Dict of metric_name -> delta (current - previous), or None if no previous.
    """
    filtered = _filter_entries(entries, policy, sample_id)

    def _day_mean(day: int, metric: str) -> Optional[float]:
        vals = [e.data.get(metric, 0) for e in filtered if e.day == day and metric in e.data]
        if not vals:
            return None
        return sum(vals) / len(vals)

    deltas: Dict[str, Optional[float]] = {}
    prev_day = current_day - 1

    for metric in _METRIC_KEYS:
        curr = _day_mean(current_day, metric)
        prev = _day_mean(prev_day, metric)
        if curr is not None and prev is not None:
            deltas[metric] = curr - prev
        else:
            deltas[metric] = None

    return deltas


def compute_summary_statistics(
    entries: List[DayLogEntry],
    policy: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute descriptive statistics (mean, std, min, max, total) per metric across all days.

    Args:
        entries: All log entries.
        policy: Optional policy filter.

    Returns:
        Dict of metric_name -> {mean, std, min, max, total}.
    """
    import statistics as stats_mod

    filtered = _filter_entries(entries, policy)
    if not filtered:
        return {}

    # Collect values per metric
    metric_values: Dict[str, List[float]] = {m: [] for m in _METRIC_KEYS}

    for entry in filtered:
        for metric in _METRIC_KEYS:
            if metric in entry.data:
                metric_values[metric].append(entry.data[metric])

    result: Dict[str, Dict[str, float]] = {}
    for metric, values in metric_values.items():
        if not values:
            continue
        result[metric] = {
            "mean": stats_mod.mean(values),
            "std": stats_mod.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "total": sum(values),
        }

    return result


def get_metric_history(
    entries: List[DayLogEntry],
    metric: str,
    policy: Optional[str] = None,
    sample_id: Optional[int] = None,
    last_n_days: int = 7,
) -> List[float]:
    """
    Get the last N days of mean values for a metric (for sparklines).

    Args:
        entries: All log entries.
        metric: The metric name.
        policy: Optional policy filter.
        sample_id: Optional sample filter.
        last_n_days: Number of recent days to include.

    Returns:
        List of mean values ordered by day (oldest first).
    """
    filtered = _filter_entries(entries, policy, sample_id)
    if not filtered:
        return []

    # Group by day
    day_values: Dict[int, List[float]] = {}
    for e in filtered:
        if metric in e.data:
            day_values.setdefault(e.day, []).append(e.data[metric])

    if not day_values:
        return []

    sorted_days = sorted(day_values.keys())
    recent_days = sorted_days[-last_n_days:]

    return [sum(day_values[d]) / len(day_values[d]) for d in recent_days]
