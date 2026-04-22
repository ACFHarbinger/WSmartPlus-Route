"""
File and pattern-based data processing utilities.

Attributes:
    process_pattern_files: Searches for and processes files matching a pattern.
    process_file: Processes a single file and modifies it based on specified criteria.

Example:
    >>> from logic.src.utils.io.file_processing import process_pattern_files, process_file
    >>> processed_count = process_pattern_files("path/to/logs")
    >>> modified = process_file("path/to/log.json")
"""

import glob
import json
import os
from typing import Any, Callable, Optional, Tuple, Union

from logic.src.interfaces.traversable import ITraversable

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

    Args:
        root_directory: The root directory to search for files.
        filename_pattern: The pattern to match files (default: "log_*.json").
        output_key: The key to extract values for (default: "km").
        process_func: Optional function to process the values (default: mean).
        update_val: Value to update the keys with (default: 0).
        input_keys: Tuple of keys to use for processing (default: (None, None)).

    Returns:
        The number of files processed.
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

    Args:
        file_path: The path to the JSON file.
        output_key: The key to extract values for (default: "km").
        process_func: Optional function to process the values (default: mean).
        update_val: Value to update the keys with (default: 0).
        input_keys: Tuple of keys to use for processing (default: (None, None)).

    Returns:
        True if the file was modified, False otherwise.
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

    checkpoint_data: object = data
    if key1 is not None and process_func is not None:
        # Two-input mode
        if isinstance(checkpoint_data, ITraversable):
            modified = process_dict_two_inputs(checkpoint_data, key1, key2, output_key, process_func)
        elif isinstance(checkpoint_data, list):
            for item in checkpoint_data:
                item_v: object = item
                if isinstance(item_v, ITraversable) and process_dict_two_inputs(
                    item_v,
                    key1,
                    key2,
                    output_key,
                    process_func,
                ):
                    modified = True
    # Single-input/Constant mode
    elif isinstance(checkpoint_data, ITraversable):
        modified = process_dict_of_dicts(checkpoint_data, output_key, process_func, update_val)
    elif isinstance(checkpoint_data, list):
        for item in checkpoint_data:
            list_item: object = item
            if isinstance(list_item, (dict, ITraversable)) and process_dict_of_dicts(
                list_item, output_key, process_func, update_val
            ):
                modified = True

    if modified:
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
        except IOError:
            return False

    return modified
