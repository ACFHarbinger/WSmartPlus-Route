"""
Data preview utilities for JSON structures.
"""

import glob
import json
import os

from .value_processing import find_single_input_values, find_two_input_values


def preview_changes(
    root_directory,
    output_key="km",
    filename_pattern="log_*.json",
    process_func=None,
    update_val=0,
    input_keys=(None, None),
):
    """
    Preview what changes will be made without actually modifying files.

    Args:
        root_directory (str): Root directory.
        output_key (str, optional): Key that would be modified.
        filename_pattern (str, optional): File pattern.
        process_func (callable, optional): Processing function.
        update_val (numeric, optional): Update value.
        input_keys (tuple, optional): (key1, key2).
    """
    assert process_func is not None, "Argument process_func must be provided"

    pattern = os.path.join(root_directory, "**", filename_pattern)
    pattern_files = glob.glob(pattern, recursive=True)
    if not pattern_files:
        print(f"No files matching '{filename_pattern}' found in {root_directory}")
        return

    # Set parameters for presence of input_keys
    key1, key2 = input_keys
    has_2_keys = key1 is not None and key2 is not None
    if has_2_keys:
        has_2_inputs = True
        key_value = key2  # Second input is the name of the second key
        input2_name = key2
    elif key1 is not None:
        has_2_inputs = True
        key_value = update_val  # Second input is the literal update_val
        input2_name = str(update_val)
    else:
        has_2_inputs = False

    print(f"Preview mode - found {len(pattern_files)} files:")
    for file_path in pattern_files:
        print(f"\nFile: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if has_2_inputs:
                # Two-input mode: find pairs of keys (input_key2 can be a string key or a value)
                key_values_found = find_two_input_values(data, input_key1=key1, input_key2=key_value)
                if key_values_found:
                    for location, value1, value2 in key_values_found:
                        new_value = process_func(value1, value2)
                        print(
                            f"- Would calculate and write to key '{output_key}' at '{location}': "
                            f"{value1} {process_func.__name__} {input2_name} → {new_value}"
                        )
                else:
                    print(f"- No suitable input pairs ({key1}, {input2_name}) found")
            else:
                # Single-input mode: find the output key
                key_values_found = find_single_input_values(data, output_key=output_key)
                if key_values_found:
                    for location, old_value in key_values_found:
                        new_value = process_func(old_value, update_val)
                        print(
                            f"- Would update file '{location}' and key '{output_key}': "
                            f"{old_value} {process_func.__name__} {update_val} → {new_value}"
                        )
                else:
                    print(f"- No '{output_key}' values found")
        except Exception as e:
            print(f"- ERROR {e}: could not read '{file_path}'")


def preview_file_changes(file_path, output_key="km", process_func=None, update_val=0, input_keys=(None, None)):
    """
    Preview changes for a single file without modifying it.

    Args:
        file_path (str): Path to the file.
        output_key (str, optional): Key to preview.
        process_func (callable, optional): Processing function.
        update_val (numeric, optional): Update value.
        input_keys (tuple, optional): (key1, key2).
    """
    assert process_func is not None, "Argument process_func must be provided"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    if not os.path.isfile(file_path):
        print(f"Path is not a file: {file_path}")
        return

    print(f"Preview mode - single file: {file_path}")

    # Set parameters for presence of input_keys
    key1, key2 = input_keys
    has_2_keys = key1 is not None and key2 is not None
    if has_2_keys:
        has_2_inputs = True
        key_value = key2  # Second input is the name of the second key
        input2_name = key2
    elif key1 is not None:
        has_2_inputs = True
        key_value = update_val  # Second input is the literal update_val
        input2_name = str(update_val)
    else:
        has_2_inputs = False

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if has_2_inputs:
            # Two-input mode: find pairs of keys
            key_values_found = find_two_input_values(data, input_key1=key1, input_key2=key_value)
            if key_values_found:
                for location, value1, value2 in key_values_found:
                    new_value = process_func(value1, value2)
                    print(
                        f"- Would calculate and write to key '{output_key}' at '{location}': "
                        f"{value1} {process_func.__name__} {input2_name} → {new_value}"
                    )
            else:
                print(f"- No suitable input pairs ({key1}, {input2_name}) found")
        else:
            # Single-input mode: find the output key
            key_values_found = find_single_input_values(data, output_key=output_key)
            if key_values_found:
                for location, old_value in key_values_found:
                    new_value = process_func(old_value, update_val)
                    print(
                        f"- Would update file '{location}' and key '{output_key}': "
                        f"{old_value} {process_func.__name__} {update_val} → {new_value}"
                    )
            else:
                print(f"- No '{output_key}' values found")
    except Exception as e:
        print(f"- ERROR {e}: could not read '{file_path}'")


def preview_pattern_files_statistics(
    root_directory,
    filename_pattern="log_*.json",
    output_filename="output.json",
    output_key="km",
    process_func=None,
):
    """
    Preview changes for pattern files statistics operation without modifying files.

    Args:
        root_directory (str): Search root.
        filename_pattern (str, optional): Input file pattern.
        output_filename (str, optional): Output filename.
        output_key (str, optional): Key to read.
        process_func (callable, optional): Processing function.
    """
    assert process_func is not None, "Argument process_func must be provided"

    # Find all matching files
    pattern = os.path.join(root_directory, "**", filename_pattern)
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"No files matching '{filename_pattern}' found in {root_directory}")
        return

    print(f"Preview mode - found {len(files)} files:")
    files_with_changes = 0
    for file_path in files:
        print(f"\nFile: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Track if any modifications would be made
            key_values_found = find_single_input_values(data, output_key=output_key)
            if key_values_found:
                # Group values by field name (the part after the last dot)
                grouped_values = {}
                for location, value in key_values_found:
                    # Extract field name from location (e.g., '[0].hexaly_vrpp0.84_gamma1' -> 'hexaly_vrpp0.84_gamma1')
                    field_name = location.split(".", 1)[-1] if "." in location else location

                    if field_name not in grouped_values:
                        grouped_values[field_name] = []
                    grouped_values[field_name].append(value)

            # Show what would be written to output file if modifications were made
            if grouped_values:
                # Get directory of current file and create output path
                file_dir = os.path.dirname(file_path)
                output_path = os.path.join(file_dir, output_filename)
                for field_name, values in grouped_values.items():
                    if values:
                        new_value = process_func(values)
                        print(
                            f"- Would write to '{output_path}': '{field_name}.{output_key}' = "
                            f"{process_func.__name__}({values}) = {new_value}"
                        )

                files_with_changes += 1
            else:
                print(f"- No '{output_key}' values found that would be processed")
        except Exception as e:
            print(f"- ERROR {e}: could not read '{file_path}'")
    print(f"\nSummary: {files_with_changes}/{len(files)} files would be processed")


def preview_file_statistics(file_path, output_filename="output.json", output_key="km", process_func=None):
    """
    Preview changes for a single file statistics operation without modifying it
    """
    assert process_func is not None, "Argument process_func must be provided"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    if not os.path.isfile(file_path):
        print(f"Path is not a file: {file_path}")
        return

    print(f"Preview mode - single file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Track if any modifications would be made
        key_values_found = find_single_input_values(data, output_key=output_key)
        if key_values_found:
            # Group values by field name (the part after the last dot)
            grouped_values = {}
            for location, value in key_values_found:
                # Extract field name from location (e.g., '[0].hexaly_vrpp0.84_gamma1' -> 'hexaly_vrpp0.84_gamma1')
                field_name = location.split(".", 1)[-1] if "." in location else location

                if field_name not in grouped_values:
                    grouped_values[field_name] = []
                grouped_values[field_name].append(value)

        # Show what would be written to output file if modifications were made
        if grouped_values:
            # Get directory of current file and create output path
            file_dir = os.path.dirname(file_path)
            output_path = os.path.join(file_dir, output_filename)
            for field_name, values in grouped_values.items():
                if values:
                    new_value = process_func(values)
                    print(
                        f"- Would write to '{output_path}': '{field_name}.{output_key}' = "
                        f"{process_func.__name__}({values}) = {new_value}"
                    )

            return True
        else:
            print(f"- No '{output_key}' values found that would be processed")
            return False
    except Exception as e:
        print(f"- ERROR {e}: could not read '{file_path}'")
