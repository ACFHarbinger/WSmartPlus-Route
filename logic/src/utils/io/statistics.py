"""
Statistical aggregation and log processing utilities.
"""

import glob
import json
import os
from typing import Callable, Dict, List, Optional, Union

from .value_processing import find_single_input_values


def process_pattern_files_statistics(
    root_directory: str,
    filename_pattern: str = "log_*.json",
    output_filename: str = "output.json",
    output_key: str = "km",
    process_func: Optional[Callable[[List[Union[int, float]]], Union[int, float]]] = None,
) -> int:
    """
    Search for log files, aggregate statistics, and write results.
    """
    processed_count = 0
    search_path = os.path.join(root_directory, "**", filename_pattern)
    files = glob.glob(search_path, recursive=True)

    for file_path in files:
        if process_file_statistics(file_path, output_filename, output_key, process_func):
            processed_count += 1

    return processed_count


def process_file_statistics(
    file_path: str,
    output_filename: str = "output.json",
    output_key: str = "km",
    process_func: Optional[Callable[[List[Union[int, float]]], Union[int, float]]] = None,
) -> bool:
    """
    Process a single log file and extract statistics.
    """
    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    # Find all occurrences of the key
    found_values = find_single_input_values(data, output_key=output_key)
    if not found_values:
        return False

    # Aggregate by parent key name
    aggregated: Dict[str, List[float]] = {}
    for path, v in found_values:
        if not isinstance(v, (int, float)):
            continue
        # Get parent key name (last component of path before the key itself)
        parent = path.split(".")[-1].split("[")[0]
        if parent not in aggregated:
            aggregated[parent] = []
        aggregated[parent].append(v)

    if not aggregated:
        return False

    output_data = {}
    for parent, values in aggregated.items():
        result = process_func(values) if process_func else sum(values) / len(values)
        output_data[parent] = {output_key: result}

    dir_path = os.path.dirname(file_path)
    out_path = os.path.join(dir_path, output_filename)

    try:
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=4)
        return True
    except IOError:
        return False
