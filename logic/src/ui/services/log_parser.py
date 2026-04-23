"""Log parsing and telemetry extraction services for simulation outputs.

This module provides specialized logic for parsing the WSmart+ Digital Twin
log format (GUI_DAY_LOG_START) from JSONL files. It handles coordinate
caching across multi-day simulation steps and aggregates numeric metrics.

Attributes:
    DayLogEntry: Dataclass representing a single day's simulation state.

Example:
    >>> from logic.src.ui.services.log_parser import parse_log_file
    >>> entries = parse_log_file(Path("simulation.jsonl"))
    >>> print(f"Parsed {len(entries)} daily records.")
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class DayLogEntry:
    """Represents a single day's simulation log entry.

    Attributes:
        policy: Identifier for the routing policy active during this day.
        sample_id: Index of the simulation instance/sample.
        day: The simulation day number (1-indexed).
        data: Raw telemetry dictionary containing performance metrics and paths.
    """

    policy: str
    sample_id: int
    day: int
    data: Dict[str, Any]


def parse_day_log_line(line: str) -> Optional[DayLogEntry]:
    """Parses a single line from the simulation log into a dataclass.

    Expected format:
        GUI_DAY_LOG_START:policy_name,sample_id,day,{json_payload}

    Args:
        line: Raw line string from the simulation JSONL log file.

    Returns:
        Optional[DayLogEntry]: The parsed entry if the line is valid, else None.
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
    """Parses all valid telemetry entries from a simulation log file.

    Includes a coordinate caching mechanism that backfills location metadata
    (all_bin_coords) for days where it might be omitted for brevity.

    Args:
        file_path: Absolute or relative path to the .jsonl log file.

    Returns:
        List[DayLogEntry]: A sequence of successfully parsed daily records.
    """
    entries: List[DayLogEntry] = []
    coords_cache = {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = parse_day_log_line(line)
                if entry is not None:
                    key = (entry.policy, entry.sample_id)
                    if "all_bin_coords" in entry.data:
                        coords_cache[key] = entry.data["all_bin_coords"]
                    elif key in coords_cache:
                        entry.data["all_bin_coords"] = coords_cache[key]

                    entries.append(entry)

            # Second pass to backfill entries that were parsed before Day 1
            for entry in entries:
                key = (entry.policy, entry.sample_id)
                if "all_bin_coords" not in entry.data and key in coords_cache:
                    entry.data["all_bin_coords"] = coords_cache[key]

    except (IOError, OSError):
        # Return empty list if file can't be read
        pass

    return entries


def stream_log_file(file_path: Path, start_line: int = 0) -> Iterator[DayLogEntry]:
    """Streams log entries from a file starting at a specific line index.

    Optimized for live monitoring, this generator maintains a coordinate cache
    to ensure interleaved telemetry remains contextually complete.

    Args:
        file_path: Path to the .jsonl log file.
        start_line: Line number to start reading from (0-indexed).

    Yields:
        Iterator[DayLogEntry]: Sequence of daily records as they are parsed.
    """
    coords_cache = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start_line:
                    # Keep coords cache warm even if skipping lines
                    if "all_bin_coords" in line:
                        entry = parse_day_log_line(line)
                        if entry is not None and "all_bin_coords" in entry.data:
                            coords_cache[(entry.policy, entry.sample_id)] = entry.data["all_bin_coords"]
                    continue

                entry = parse_day_log_line(line)
                if entry is not None:
                    key = (entry.policy, entry.sample_id)
                    if "all_bin_coords" in entry.data:
                        coords_cache[key] = entry.data["all_bin_coords"]
                    elif key in coords_cache:
                        entry.data["all_bin_coords"] = coords_cache[key]

                    yield entry
    except (IOError, OSError):
        return


def get_unique_policies(entries: List[DayLogEntry]) -> List[str]:
    """Extracts a sorted list of unique policy names from telemetry.

    Args:
        entries: Master list of log entries to analyze.

    Returns:
        List[str]: Alphabetically sorted unique policy identifiers.
    """
    return sorted(set(e.policy for e in entries))


def get_unique_samples(entries: List[DayLogEntry]) -> List[int]:
    """Extracts a sorted list of unique simulation sample IDs from telemetry.

    Args:
        entries: Master list of log entries to analyze.

    Returns:
        List[int]: Numerically sorted unique sample indices.
    """
    return sorted(set(e.sample_id for e in entries))


def filter_entries(
    entries: List[DayLogEntry],
    policy: Optional[str] = None,
    sample_id: Optional[int] = None,
    day: Optional[int] = None,
) -> List[DayLogEntry]:
    """Filters log entries based on policy, sample, and day criteria.

    Args:
        entries: Master list of log entries to filter.
        policy: Exact policy name to match.
        sample_id: Exact sample ID to match.
        day: Exact day number to match.

    Returns:
        List[DayLogEntry]: The subset of entries matching all specified criteria.
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
    """Retrieves the global minimum and maximum day numbers from telemetry.

    Args:
        entries: Master list of simulation log entries.

    Returns:
        Tuple[int, int]: (min_day, max_day).
    """
    if not entries:
        return (0, 0)
    days = [e.day for e in entries]
    return (min(days), max(days))


def aggregate_metrics_by_day(
    entries: List[DayLogEntry], policy: Optional[str] = None
) -> Dict[int, Dict[str, List[float]]]:
    """Aggregates numeric metrics across samples for each simulation day.

    Useful for tracking convergence and stability across multiple runs.

    Args:
        entries: Master list of log entries.
        policy: Optional filter to restrict aggregation to one policy.

    Returns:
        Dict[int, Dict[str, List[float]]]: Day -> metric -> list of sample values.
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
