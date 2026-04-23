"""Data loading and aggregation services for benchmark results.

This module provides specialized logic for parsing high-performance
benchmarking logs (JSONL) and transforming them into analysis-ready
DataFrames. It supports performance metrics, latency tracking, and
throughput comparisons across hardware levels.

Attributes:
    load_benchmark_data: Parses the main benchmark log file into a DataFrame.
    get_unique_benchmarks: Extracts sorted list of benchmark identifiers.

Example:
    >>> from logic.src.ui.services.benchmark_loader import load_benchmark_data
    >>> df = load_benchmark_data()
    >>> print(f"Loaded {len(df)} benchmark entries.")
"""

import json
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st


def get_benchmark_log_path() -> Path:
    """Retrieves the absolute path to the benchmark log file.

    Returns:
        Path: The filesystem path to benchmarks.jsonl.
    """
    return Path("logs/benchmarks/benchmarks.jsonl")


@st.cache_data(ttl=30)
def load_benchmark_data() -> pd.DataFrame:
    """Parses benchmark results from the JSONL log file into a DataFrame.

    Filters specifically for 'performance_benchmark' records and flattens
    associated metrics and metadata for easier analysis.

    Returns:
        pd.DataFrame: A DataFrame sorted by timestamp (descending).
    """
    log_path = get_benchmark_log_path()
    if not log_path.exists():
        return pd.DataFrame()

    entries = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Filter for performance benchmarks
                    if entry.get("type") != "performance_benchmark":
                        continue

                    # Flatten metrics and metadata
                    flattened = {
                        "timestamp": entry.get("timestamp"),
                        "benchmark": entry.get("benchmark"),
                        "level": entry.get("level"),
                        "message": entry.get("message"),
                    }
                    if "metrics" in entry:
                        flattened.update(entry["metrics"])
                    if "metadata" in entry:
                        flattened.update(entry["metadata"])
                    entries.append(flattened)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return pd.DataFrame()

    if not entries:
        return pd.DataFrame()

    df = pd.DataFrame(entries)
    if not df.empty and "timestamp" in df.columns:
        # ISO string timestamps sort correctly
        df = df.sort_values("timestamp", ascending=False)

    return df


def get_unique_benchmarks(df: pd.DataFrame) -> List[str]:
    """Extracts a sorted list of unique benchmark identifiers from the data.

    Args:
        df: The benchmark results DataFrame.

    Returns:
        List[str]: Unique benchmark names found in the 'benchmark' column.
    """
    if df.empty or "benchmark" not in df.columns:
        return []
    return sorted(df["benchmark"].unique().tolist())
