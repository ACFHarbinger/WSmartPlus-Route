"""Simulation analytics and statistical computation services.

This module provides specialized logic for computing aggregate performance
metrics across multi-day simulation runs. It supports cumulative totals,
day-over-day delta calculation, and descriptive statistics (mean, std,
min, max) for all primary performance indicators.

Attributes:
    compute_cumulative_stats: Computes totals across all days.
    compute_day_deltas: Computes change from previous day.
    compute_summary_statistics: Computes mean, std, etc. per metric.
    get_metric_history: Returns historical mean values.

Example:
    >>> from logic.src.ui.services.simulation_analytics import compute_cumulative_stats
    >>> totals = compute_cumulative_stats(log_entries, policy="greedy")
    >>> print(f"Total distance: {totals['Total Distance (km)']:.2f} km")
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
    """Internal helper to filter entries by policy and sample ID.

    Args:
        entries: Master list of log entries.
        policy: Optional filter for policy identifier.
        sample_id: Optional filter for simulation sample index.

    Returns:
        List[DayLogEntry]: The filtered subset of entries.
    """
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
    """Computes cumulative totals across all days for a policy and sample.

    Args:
        entries: Master list of log entries.
        policy: Optional filter for policy identifier.
        sample_id: Optional filter for simulation sample index.

    Returns:
        Dict[str, float]: Mapping of 'Total <Metric>' to its cumulative value.
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
    """Computes metric deltas between the current day and the previous day.

    Args:
        entries: Master list of log entries.
        current_day: The simulation day for which to compute deltas.
        policy: Optional filter for policy identifier.
        sample_id: Optional filter for simulation sample index.

    Returns:
        Dict[str, Optional[float]]: Metric names mapped to their day-over-day change.
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
    """Computes aggregate descriptive statistics across all simulation days.

    Args:
        entries: Master list of log entries.
        policy: Optional filter for policy identifier.

    Returns:
        Dict[str, Dict[str, float]]: Metric names mapped to {mean, std, min, max, total}.
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
    """Gets the last N days of mean values for a metric (used for sparklines).

    Args:
        entries: Master list of log entries.
        metric: The specific metric identifier to track.
        policy: Optional filter for policy identifier.
        sample_id: Optional filter for simulation sample index.
        last_n_days: Number of historical days to retrieve. Defaults to 7.

    Returns:
        List[float]: Sequence of daily mean values (oldest first).
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
