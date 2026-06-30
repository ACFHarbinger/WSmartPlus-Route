"""
I/O utilities package.

Attributes:
    process_dict_of_dicts: Process a dictionary of dictionaries and apply a function to key values.
    process_list_of_dicts: Process a list of dictionaries.
    process_dict_two_inputs: Process a dictionary using two input keys for a transformation.
    process_list_two_inputs: Process a list of dictionaries using two input keys.
    process_file: Processes a single file and modifies it based on specified criteria.
    process_pattern_files: Searches for and processes files matching a pattern.
    read_json: Read JSON data from a file.
    zip_directory: Zip a directory.
    extract_zip: Extract a zip file.
    confirm_proceed: Confirm proceeding with an action.
    compose_dirpath: Compose directory path.
    read_output: Read output from a file.
    preview_changes: Preview changes to a file.
    preview_file_changes: Preview changes to a file.
    preview_pattern_files_statistics: Preview statistics for files matching a pattern.
    preview_file_statistics: Preview statistics for a single file.
    split_file: Split a file.
    chunk_zip_content: Chunk zip content.
    reassemble_files: Reassemble files.
    process_file_statistics: Calculates statistics for a single file.
    process_pattern_files_statistics: Calculates statistics for files matching a pattern.
    find_single_input_values: Find values for a single input key in a nested structure.
    find_two_input_values: Find values for two input keys in a nested structure.

Example:
    >>> from logic.src.utils.io import find_single_input_values, find_two_input_values, process_dict_of_dicts, process_list_of_dicts, process_dict_two_inputs, process_list_two_inputs, process_file, process_pattern_files, process_file_statistics, process_pattern_files_statistics
    >>> single_values = find_single_input_values(data, "key")
    >>> two_values = find_two_input_values(data, "key1", "key2")
    >>> processed = process_dict_of_dicts(data, "km", lambda x, y: x + y, 10)
    >>> processed = process_list_of_dicts(data, "km", lambda x, y: x + y, 10)
    >>> processed = process_dict_two_inputs(data, "key1", "key2", "output_key", lambda x, y: x + y)
    >>> processed = process_list_two_inputs(data, "key1", "key2", "output_key", lambda x, y: x + y)
    >>> modified = process_file("path/to/log.json")
    >>> processed_count = process_pattern_files("path/to/logs")
    >>> stats = process_file_statistics("path/to/log.json")
    >>> pattern_stats = process_pattern_files_statistics("path/to/logs")
"""

from .dict_processing import (
    process_dict_of_dicts,
    process_dict_two_inputs,
    process_list_of_dicts,
    process_list_two_inputs,
)
from .file_processing import process_file, process_pattern_files
from .files import compose_dirpath, confirm_proceed, extract_zip, read_json, zip_directory
from .locking import read_output
from .preview import preview_changes, preview_file_changes, preview_file_statistics, preview_pattern_files_statistics
from .splitting import chunk_zip_content, reassemble_files, split_file
from .statistics import process_file_statistics, process_pattern_files_statistics
from .value_processing import find_single_input_values, find_two_input_values

__all__ = [
    "process_dict_of_dicts",
    "process_dict_two_inputs",
    "process_list_of_dicts",
    "process_list_two_inputs",
    "process_file",
    "process_pattern_files",
    "read_json",
    "zip_directory",
    "extract_zip",
    "confirm_proceed",
    "compose_dirpath",
    "read_output",
    "preview_changes",
    "preview_file_changes",
    "preview_pattern_files_statistics",
    "preview_file_statistics",
    "split_file",
    "chunk_zip_content",
    "reassemble_files",
    "process_file_statistics",
    "process_pattern_files_statistics",
    "find_single_input_values",
    "find_two_input_values",
]
