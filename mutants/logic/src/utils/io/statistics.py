"""
Statistical aggregation and log processing utilities.
"""

import glob
import json
import os
from typing import Callable, List, Optional, Union

from .values_processing import find_single_input_values


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

    # Extract just numeric values
    numeric_values = [v for _, v in found_values if isinstance(v, (int, float))]
    if not numeric_values:
        return False

    # Aggregate
    if process_func:
        result = process_func(numeric_values)
    else:
        # Default: Mean
        result = sum(numeric_values) / len(numeric_values)

    output_data = {
        "source": file_path,
        "key": output_key,
        "count": len(numeric_values),
        "result": result,
    }

    dir_path = os.path.dirname(file_path)
    out_path = os.path.join(dir_path, output_filename)

    try:
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=4)
        return True
    except IOError:
        return False
