"""Data loading and caching services for the Streamlit dashboard.

This module provides high-level utilities for discovering and loading
training logs (Lightning CSVs) and simulation telemetry (JSONL files).
It utilizes Streamlit's caching mechanisms to ensure responsive UI
interactions even with large datasets.

Attributes:
    discover_training_runs: Scans logs/output for versioned training folders.
    load_hparams: Reads hyperparameter YAML files for a given run.
    load_training_metrics: Loads Lightning metrics into a Pandas DataFrame.
    load_policy_params: Retrieves policy configuration from tracking DB.

Example:
    >>> from logic.src.ui.services.data_loader import discover_training_runs
    >>> runs = discover_training_runs()
    >>> print(f"Found {len(runs)} training versions.")
"""

import json
import os
import sqlite3
import statistics
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
    """Retrieves the absolute path to the project root directory.

    Returns:
        Path: The validated project root directory.
    """
    return Path.cwd()


def get_logs_dir() -> Path:
    """Retrieves the absolute path to the training logs directory.

    Returns:
        Path: The directory where model checkpoints and metrics are stored.
    """
    return get_project_root() / "logs"


def get_simulation_output_dir() -> Path:
    """Retrieves the absolute path to the simulation output directory.

    Returns:
        Path: The directory containing simulation JSONL logs.
    """
    return get_project_root() / "assets" / "output"


# -----------------------------------------------------------------------------
# Training Logs (Lightning)
# -----------------------------------------------------------------------------


@st.cache_data(ttl=60)
def discover_training_runs() -> List[Tuple[str, Path]]:
    """Scans the designated logs directory for valid training runs.

    Returns:
        List[Tuple[str, Path]]: A sequence of (version_name, metrics_csv_path).
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
    """Loads hyperparameters for a specific Lightning training version.

    Args:
        version_name: Identifier for the version (e.g., "version_0").

    Returns:
        Dict[str, Any]: Mapping of hyperparameter keys to values.
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
    """Loads training performance metrics from a CSV file.

    Args:
        metrics_path: Absolute path to the metrics.csv file.

    Returns:
        pd.DataFrame: A DataFrame indexed by epoch/step with metric columns.
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
    """Retrieves and merges metrics from multiple training runs.

    Args:
        version_names: List of version identifiers (e.g., ["version_0", "version_1"]).

    Returns:
        Dict[str, pd.DataFrame]: Mapping of version name to its metrics DataFrame.
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
    """Retrieves structured policy configuration from the tracking database.

    Args:
        policy_name: Identifier for the routing policy.
        sample_id: index of the simulation sample.

    Returns:
        Dict[str, Any]: Mapping of configuration parameter names to values.
    """
    db_path = get_project_root() / "assets" / "tracking" / "tracking.db"
    if not db_path.exists():
        return {}

    prefix = f"policy_params/{policy_name}/s{sample_id}/"
    params: Dict[str, Any] = {}

    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row

        # 1. Dynamically load the SQL query from the external file
        query_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_loader_query.sql")
        with open(query_path, "r", encoding="utf-8") as f:
            query = f.read()

        # 2. Execute the loaded query
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
    """Identifies all simulation telemetry (JSONL) files in the output directory.

    Returns:
        List[Tuple[str, Path]]: Sequence of (display_name, absolute_path).
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
    """Loads and parses a simulation JSONL file with caching enabled.

    Args:
        log_path: Absolute filesystem path to the log file.

    Returns:
        List[Dict[str, Any]]: Sequence of daily telemetry records as dictionaries.
    """
    path = Path(log_path)
    entries = parse_log_file(path)

    # Convert to dicts for caching (dataclasses may not serialize well)
    return [{"policy": e.policy, "sample_id": e.sample_id, "day": e.day, "data": e.data} for e in entries]


def load_simulation_log_fresh(log_path: str) -> List[DayLogEntry]:
    """Loads a simulation log without caching, suitable for live tailing.

    Args:
        log_path: Absolute filesystem path to the log file.

    Returns:
        List[DayLogEntry]: Sequence of parsed daily record dataclasses.
    """
    return parse_log_file(Path(log_path))


def get_simulation_metadata(entries: List[DayLogEntry]) -> Dict[str, Any]:
    """Extracts summary metadata (policies, samples, day range) from telemetry.

    Args:
        entries: Master list of parsed log entries.

    Returns:
        Dict[str, Any]: Mapping with 'policies', 'samples', and 'day_range'.
    """
    return {
        "policies": get_unique_policies(entries),
        "samples": get_unique_samples(entries),
        "day_range": get_day_range(entries),
    }


def entries_to_dataframe(entries: List[DayLogEntry]) -> pd.DataFrame:
    """Converts dataclass-style log entries into a flattened Pandas DataFrame.

    Args:
        entries: Master list of DayLogEntry dataclasses.

    Returns:
        pd.DataFrame: A long-form DataFrame with metric columns.
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
    """Computes daily averages and standard deviations for all metrics.

    Args:
        entries: Master list of log entries.
        policy: Optional filter to restrict analysis to a single policy.

    Returns:
        pd.DataFrame: Statistics indexed by day with [metric]_mean and [metric]_std.
    """
    aggregated = aggregate_metrics_by_day(entries, policy)

    rows = []
    for day in sorted(aggregated.keys()):
        metrics = aggregated[day]
        row: Dict[str, Any] = {"day": day}
        for metric, values in metrics.items():
            if values:
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)
