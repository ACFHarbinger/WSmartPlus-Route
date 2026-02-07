"""
File and pattern-based data processing utilities.
"""

import glob
import json
import os
from typing import Any, Callable, Optional, Tuple, Union

from .dict_processing import process_dict_of_dicts, process_dict_two_inputs


def process_pattern_files(
    root_directory: str,
    filename_pattern: str = "log_*.json",
    output_key: str = "km",
    process_func: Optional[Callable[[Any, Any], Any]] = None,
    update_val: Union[int, float] = 0,
    input_keys: Tuple[Optional[str], Union[str, int, float, None]] = (None, None),
) -> int:
    """
    Search for and process files matching a pattern.

    Returns:
        Number of modified files.
    """
    modified_count = 0
    search_path = os.path.join(root_directory, "**", filename_pattern)
    files = glob.glob(search_path, recursive=True)

    for file_path in files:
        if process_file(file_path, output_key, process_func, update_val, input_keys):
            modified_count += 1

    return modified_count


def process_file(
    file_path: str,
    output_key: str = "km",
    process_func: Optional[Callable[[Any, Any], Any]] = None,
    update_val: Union[int, float] = 0,
    input_keys: Tuple[Optional[str], Union[str, int, float, None]] = (None, None),
) -> bool:
    """
    Modify a single JSON file.
    """
    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    modified = False
    key1, key2 = input_keys

    if key1 is not None and process_func is not None:
        # Two-input mode
        if isinstance(data, dict):
            modified = process_dict_two_inputs(data, key1, key2, output_key, process_func)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if process_dict_two_inputs(item, key1, key2, output_key, process_func):
                        modified = True
    else:
        # Single-input/Constant mode
        if isinstance(data, dict):
            modified = process_dict_of_dicts(data, output_key, process_func, update_val)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if process_dict_of_dicts(item, output_key, process_func, update_val):
                        modified = True

    if modified:
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
        except IOError:
            return False

    return modified
