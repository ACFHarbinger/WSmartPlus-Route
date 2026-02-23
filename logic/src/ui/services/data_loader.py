# Copyright (c) WSmart-Route. All rights reserved.
"""
Data loading and caching services for the dashboard.

Provides cached access to training logs and simulation outputs.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml

from logic.src.ui.services.log_parser import (
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
def load_policy_params(policy_name: str, sample_id: int) -> Dict[str, Any]:
    """Retrieve structured policy configuration from the tracking database.

    Args:
        policy_name: Name of the routing policy.
        sample_id: Index of the simulation instance.

    Returns:
        Dict of parameters matching the prefix 'policy_params/{policy_name}/s{sample_id}/'.
    """
    db_path = get_project_root() / "assets" / "tracking" / "tracking.db"
    if not db_path.exists():
        return {}

    prefix = f"policy_params/{policy_name}/s{sample_id}/"
    params: Dict[str, Any] = {}

    try:
        # We use raw sqlite3 here to avoid adding a heavy dependency on TrackingStore if not needed,
        # but following the prefix-matching logic for policy_params.
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row

        # Get params from the latest run that has these keys
        query = """
            SELECT key, value
            FROM params
            WHERE key LIKE ?
            ORDER BY id DESC
        """
        rows = conn.execute(query, (f"{prefix}%",)).fetchall()

        for row in rows:
            key = row["key"]
            val_raw = row["value"]

            # Clean key
            p_key = key[len(prefix) :]

            # If we already have this key from a newer run (due to ORDER BY id DESC), skip
            if p_key in params:
                continue

            # Try to parse JSON if it looks like a stringified list/dict
            try:
                if isinstance(val_raw, str) and (val_raw.startswith("{") or val_raw.startswith("[")):
                    params[p_key] = json.loads(val_raw)
                else:
                    params[p_key] = val_raw
            except Exception:
                params[p_key] = val_raw

        conn.close()
    except Exception as e:
        st.warning(f"Failed to load params from database: {e}")
        return {}

    return params


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
# Re-exports from simulation_analytics (backward compatibility)
# -----------------------------------------------------------------------------

from logic.src.ui.services.simulation_analytics import (  # noqa: F401, E402
    compute_cumulative_stats,
    compute_day_deltas,
    compute_summary_statistics,
    get_metric_history,
)
