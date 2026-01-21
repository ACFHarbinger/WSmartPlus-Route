"""
Input/Output utilities facade.
"""
from __future__ import annotations

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
from .io.processing import (
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
