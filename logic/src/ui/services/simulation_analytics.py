"""
Simulation analytics functions for the dashboard.

Extracted from ``data_loader.py`` to keep module sizes under 400 LoC.
Functions are re-exported from ``data_loader.py`` for backward compatibility.
"""

import statistics as stats_mod
from typing import Dict, List, Optional

from logic.src.ui.services.log_parser import DayLogEntry

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
