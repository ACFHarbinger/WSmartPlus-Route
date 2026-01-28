# Copyright (c) WSmart-Route. All rights reserved.
"""
Custom log parsing logic for simulation outputs.

Parses the custom GUI_DAY_LOG_START format from JSONL files.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class DayLogEntry:
    """Represents a single day's simulation log entry."""

    policy: str
    sample_id: int
    day: int
    data: Dict[str, Any]


def parse_day_log_line(line: str) -> Optional[DayLogEntry]:
    """
    Parse a single line from the simulation log.

    Expected format:
        GUI_DAY_LOG_START:policy_name,sample_id,day,{json_payload}

    Args:
        line: Raw line from the log file.

    Returns:
        DayLogEntry if parsing succeeds, None otherwise.
    """
    line = line.strip()
    if not line.startswith("GUI_DAY_LOG_START:"):
        return None

    try:
        # Remove the prefix
        content = line[len("GUI_DAY_LOG_START:") :]

        # Split by the first 3 commas to extract metadata
        parts = content.split(",", 3)
        if len(parts) < 4:
            return None

        policy = parts[0].strip()
        sample_id = int(parts[1].strip())
        day = int(parts[2].strip())
        json_str = parts[3].strip()

        # Parse the JSON payload
        data = json.loads(json_str)

        return DayLogEntry(policy=policy, sample_id=sample_id, day=day, data=data)

    except (ValueError, json.JSONDecodeError):
        # Handle partial lines or malformed JSON gracefully
        return None


def parse_log_file(file_path: Path) -> List[DayLogEntry]:
    """
    Parse all valid entries from a simulation log file.

    Args:
        file_path: Path to the .jsonl log file.

    Returns:
        List of parsed DayLogEntry objects.
    """
    entries: List[DayLogEntry] = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = parse_day_log_line(line)
                if entry is not None:
                    entries.append(entry)
    except (IOError, OSError):
        # Return empty list if file can't be read
        pass

    return entries


def stream_log_file(file_path: Path, start_line: int = 0) -> Iterator[DayLogEntry]:
    """
    Stream log entries from a file, starting from a specific line.

    Useful for tailing log files in live mode.

    Args:
        file_path: Path to the .jsonl log file.
        start_line: Line number to start from (0-indexed).

    Yields:
        DayLogEntry objects as they are parsed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start_line:
                    continue
                entry = parse_day_log_line(line)
                if entry is not None:
                    yield entry
    except (IOError, OSError):
        return


def get_unique_policies(entries: List[DayLogEntry]) -> List[str]:
    """Extract unique policy names from log entries."""
    return sorted(set(e.policy for e in entries))


def get_unique_samples(entries: List[DayLogEntry]) -> List[int]:
    """Extract unique sample IDs from log entries."""
    return sorted(set(e.sample_id for e in entries))


def filter_entries(
    entries: List[DayLogEntry],
    policy: Optional[str] = None,
    sample_id: Optional[int] = None,
    day: Optional[int] = None,
) -> List[DayLogEntry]:
    """
    Filter log entries by criteria.

    Args:
        entries: List of log entries to filter.
        policy: Filter by policy name (optional).
        sample_id: Filter by sample ID (optional).
        day: Filter by day number (optional).

    Returns:
        Filtered list of entries.
    """
    result = entries

    if policy is not None:
        result = [e for e in result if e.policy == policy]
    if sample_id is not None:
        result = [e for e in result if e.sample_id == sample_id]
    if day is not None:
        result = [e for e in result if e.day == day]

    return result


def get_day_range(entries: List[DayLogEntry]) -> Tuple[int, int]:
    """Get the min and max day numbers from entries."""
    if not entries:
        return (0, 0)
    days = [e.day for e in entries]
    return (min(days), max(days))


def aggregate_metrics_by_day(
    entries: List[DayLogEntry], policy: Optional[str] = None
) -> Dict[int, Dict[str, List[float]]]:
    """
    Aggregate metrics across samples for each day.

    Args:
        entries: List of log entries.
        policy: Optional policy filter.

    Returns:
        Dict mapping day -> metric_name -> list of values across samples.
    """
    filtered = filter_entries(entries, policy=policy)
    result: Dict[int, Dict[str, List[float]]] = {}

    numeric_metrics = ["overflows", "kg", "km", "kg/km", "cost", "profit", "ncol", "kg_lost"]

    for entry in filtered:
        day = entry.day
        if day not in result:
            result[day] = {m: [] for m in numeric_metrics}

        for metric in numeric_metrics:
            if metric in entry.data:
                value = entry.data[metric]
                if isinstance(value, (int, float)):
                    result[day][metric].append(float(value))

    return result
