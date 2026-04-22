"""
Input/Output utilities facade.

Attributes:
    read_json: Read JSON file.
    zip_directory: Zip directory.
    extract_zip: Extract zip file.
    confirm_proceed: Confirm proceed.
    compose_dirpath: Compose directory path.
    split_file: Split file.
    chunk_zip_content: Chunk zip content.
    reassemble_files: Reassemble files.
    process_dict_of_dicts: Process dict of dicts.
    process_list_of_dicts: Process list of dicts.
    process_dict_two_inputs: Process dict two inputs.
    process_list_two_inputs: Process list two inputs.
    find_single_input_values: Find single input values.
    find_two_input_values: Find two input values.
    process_pattern_files: Process pattern files.
    process_file: Process file.
    process_pattern_files_statistics: Process pattern files statistics.
    process_file_statistics: Process file statistics.
    preview_changes: Preview changes.
    preview_file_changes: Preview file changes.
    preview_pattern_files_statistics: Preview pattern files statistics.
    preview_file_statistics: Preview file statistics.
    read_output: Read output.

Example:
    >>> from logic.src.utils.io_utils import read_json
    >>> data = read_json("data.json")
    >>> zip_directory("data", "data.zip")
    >>> extract_zip("data.zip", "data")
    >>> confirm_proceed("Are you sure?")
    True
    >>> compose_dirpath("data", "data.json")
    "data/data.json"
    >>> split_file("data.json", 10)
    >>> chunk_zip_content("data.zip", 10)
    >>> reassemble_files(["data1.json", "data2.json"])
    >>> process_dict_of_dicts({"a": {"b": 1}})
    {"a": {"b": 1}}
    >>> process_list_of_dicts([{"a": 1}, {"a": 2}])
    [{"a": 1}, {"a": 2}]
    >>> process_dict_two_inputs({"a": {"b": 1}, "c": {"d": 2}})
    {"a": {"b": 1}, "c": {"d": 2}}
    >>> process_list_two_inputs([{"a": 1}, {"a": 2}], [{"b": 3}, {"b": 4}])
    [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
    >>> find_single_input_values([{"a": 1}, {"a": 2}])
    [1, 2]
    >>> find_two_input_values([{"a": 1}, {"a": 2}], [{"b": 3}, {"b": 4}])
    [(1, 3), (2, 4)]
    >>> process_pattern_files("data/*.json")
    >>> process_file("data.json")
    >>> process_pattern_files_statistics("data/*.json")
    >>> process_file_statistics("data.json")
    >>> preview_changes([{"a": 1}, {"a": 2}])
    >>> preview_file_changes("data.json")
    >>> preview_pattern_files_statistics("data/*.json")
    >>> preview_file_statistics("data.json")
    >>> read_output("output.json")
    {"a": 1}
"""

from __future__ import annotations

from .io import (
    find_single_input_values,
    find_two_input_values,
    process_dict_of_dicts,
    process_dict_two_inputs,
    process_file,
    process_file_statistics,
    process_list_of_dicts,
    process_list_two_inputs,
    process_pattern_files,
    process_pattern_files_statistics,
)

# Re-export from new modules
from .io.files import (
    compose_dirpath,
    confirm_proceed,
    extract_zip,
    read_json,
    zip_directory,
)
from .io.locking import read_output
from .io.preview import (
    preview_changes,
    preview_file_changes,
    preview_file_statistics,
    preview_pattern_files_statistics,
)
from .io.splitting import (
    chunk_zip_content,
    reassemble_files,
    split_file,
)

__all__ = [
    "read_json",
    "zip_directory",
    "extract_zip",
    "confirm_proceed",
    "compose_dirpath",
    "split_file",
    "chunk_zip_content",
    "reassemble_files",
    "process_dict_of_dicts",
    "process_list_of_dicts",
    "process_dict_two_inputs",
    "process_list_two_inputs",
    "find_single_input_values",
    "find_two_input_values",
    "process_pattern_files",
    "process_file",
    "process_pattern_files_statistics",
    "process_file_statistics",
    "preview_changes",
    "preview_file_changes",
    "preview_pattern_files_statistics",
    "preview_file_statistics",
    "read_output",
]
