# Copyright (c) WSmart-Route. All rights reserved.
"""
Data loading service for benchmark results.
"""

import json
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st


def get_benchmark_log_path() -> Path:
    """Get the path to the benchmark log file."""
    return Path("logs/benchmarks.jsonl")


@st.cache_data(ttl=30)
def load_benchmark_data() -> pd.DataFrame:
    """
    Load benchmark results from JSONL file.

    Returns:
        DataFrame with flattened benchmark entries.
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
    """Get unique benchmark names from the data."""
    if df.empty or "benchmark" not in df.columns:
        return []
    return sorted(df["benchmark"].unique().tolist())
