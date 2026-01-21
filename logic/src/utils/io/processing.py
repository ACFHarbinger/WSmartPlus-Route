"""
Data processing utilities for JSON structures.
"""

import glob
import json
import os


def process_dict_of_dicts(data_dict, output_key="km", process_func=None, update_val=0):
    """
    Process a dictionary of dictionaries and apply process_func to key values.
    Supports single numeric values OR lists of numeric values.

    Args:
        data_dict (dict): The dictionary to process.
        output_key (str, optional): The key within inner dicts to modify. Defaults to 'km'.
        process_func (callable, optional): function(old_val, update_val) -> new_val.
        update_val (int/float, optional): The second argument for process_func. Defaults to 0.

    Returns:
        bool: True if any modification was made, False otherwise.
    """
    assert process_func is not None, "Argument process_func must be provided"

    modified = False
    for key, value in data_dict.items():
        if isinstance(value, dict) and output_key in value:
            target_value = value[output_key]

            if isinstance(target_value, (int, float)):
                # Case 1: Single numeric value (Original logic)
                modified = True
                old_val = target_value
                value[output_key] = process_func(old_val, update_val)
                print(f"   -- Updated '{output_key}' in '{key}': {old_val} → {value[output_key]}")

            elif isinstance(target_value, list):
                # Case 2: List of numeric values (New logic)
                list_modified = False
                for i, item in enumerate(target_value):
                    if isinstance(item, (int, float)):
                        list_modified = True
                        old_val = item
                        target_value[i] = process_func(old_val, update_val)
                        print(f"   -- Updated '{output_key}[{i}]' in '{key}': {old_val} → {target_value[i]}")

                if list_modified:
                    modified = True
                # No 'else' print here, as an empty list is valid and doesn't warrant a warning.

            else:
                # Handle non-list, non-numeric values
                print(f"   -- Warning: '{output_key}' value is not numeric or a list in '{key}'")

    return modified


def process_list_of_dicts(data_list, output_key="km", process_func=None, update_val=0):
    """
    Process a list of dictionaries of dictionaries.

    Args:
        data_list (list): The list of dictionaries to process.
        output_key (str, optional): The key to modify. Defaults to 'km'.
        process_func (callable, optional): The function to apply.
        update_val (int/float, optional): The update value. Defaults to 0.

    Returns:
        bool: True if any modification was made.
    """
    assert process_func is not None, "Argument process_func must be provided"

    modified = False
    for item in data_list:
        if isinstance(item, dict):
            # Recursively process each dictionary in the list
            if process_dict_of_dicts(item, output_key, process_func, update_val):
                modified = True
    return modified


def process_dict_two_inputs(data_dict, input_key1, input_key2_or_val, output_key, process_func):
    """
    Process a dictionary of dictionaries and apply process_func using two inputs.
    Format: [output_key] = process_func([input_key1], [input_key2_or_val])

    Args:
        data_dict (dict): The dictionary to process.
        input_key1 (str): key for the first input operand.
        input_key2_or_val (str or numeric): key for second input, or a literal value.
        output_key (str): key to store the result.
        process_func (callable): function(val1, val2) -> new_val.

    Returns:
        bool: True if any modification was made.
    """
    assert process_func is not None, "Argument process_func must be provided"

    modified = False
    is_two_keys = isinstance(input_key2_or_val, str)
    for key, value_dict in data_dict.items():
        if isinstance(value_dict, dict) and input_key1 in value_dict:
            input_val1 = value_dict[input_key1]
            input_val2 = None

            # Determine second input value
            if is_two_keys:
                # Case: Two input keys (input_key2_or_val is the second key name)
                if input_key2_or_val in value_dict:
                    input_val2 = value_dict[input_key2_or_val]
            else:
                # Case: One input key + one literal value (input_key2_or_val is the literal value)
                input_val2 = input_key2_or_val

            # --- Perform Calculation ---
            if input_val2 is not None:
                if isinstance(input_val1, (int, float)) and isinstance(input_val2, (int, float)):
                    # Check for single numeric value for input 1 and 2
                    if process_func.__name__ == "/" and input_val2 == 0:
                        print(f"   -- WARNING: Skipping division by zero in '{key}' (Input 2 is zero)")
                        continue

                    new_val = process_func(input_val1, input_val2)
                    value_dict[output_key] = new_val
                    modified = True
                    input2_name = input_key2_or_val if is_two_keys else "value"
                    print(
                        f"   -- Calculated '{output_key}' in '{key}' "
                        f"({input_key1} {process_func.__name__} {input2_name}): {new_val}"
                    )
                elif (
                    isinstance(input_val1, list) and isinstance(input_val2, list) and len(input_val1) == len(input_val2)
                ):
                    # Check for list of numeric values for both inputs
                    new_list = []
                    list_modified = False
                    input2_name = input_key2_or_val if is_two_keys else "value"
                    for i in range(len(input_val1)):
                        item1 = input_val1[i]
                        item2 = input_val2[i]
                        if isinstance(item1, (int, float)) and isinstance(item2, (int, float)):
                            if process_func.__name__ == "/" and item2 == 0:
                                print(f"   -- WARNING: Skipping division by zero for list item {i} in '{key}'")
                                # Try to keep the original output value if it exists, otherwise use None
                                new_list.append(
                                    value_dict.get(output_key, [])[i]
                                    if output_key in value_dict and i < len(value_dict[output_key])
                                    else None
                                )
                                continue

                            new_item = process_func(item1, item2)
                            new_list.append(new_item)
                            list_modified = True
                            print(
                                f"   -- Calculated list item {i} of '{output_key}' in '{key}' "
                                f"({input_key1}[{i}] {process_func.__name__} {input2_name}[{i}]): {new_item}"
                            )
                        else:
                            # If non-numeric, try to keep the original output value if it exists
                            new_list.append(
                                value_dict.get(output_key, [])[i]
                                if output_key in value_dict and i < len(value_dict[output_key])
                                else None
                            )

                    if list_modified:
                        value_dict[output_key] = new_list
                        modified = True
                else:
                    print(
                        f"   -- Warning: Input values for '{input_key1}' and '{input_key2_or_val}' "
                        f"are not compatible numeric types or lists in '{key}'"
                    )
    return modified


def process_list_two_inputs(data_list, input_key1, input_key2_or_val, output_key, process_func):
    """
    Process a list of dictionaries of dictionaries using two inputs.

    Args:
        data_list (list): The list of dictionaries.
        input_key1 (str): First input key.
        input_key2_or_val (str or numeric): Second input key or value.
        output_key (str): Output key.
        process_func (callable): Processing function.

    Returns:
        bool: True if modified.
    """
    assert process_func is not None, "Argument process_func must be provided"

    modified = False
    for item in data_list:
        if isinstance(item, dict):
            # Recursively process each dictionary in the list
            if process_dict_two_inputs(item, input_key1, input_key2_or_val, output_key, process_func):
                modified = True
    return modified


def find_single_input_values(data, current_path="", output_key="km"):
    """
    Recursively find all *source* values for single-input mode.

    Args:
        data (dict or list): The data structure to search.
        current_path (str, optional): Current path in the recursive search.
        output_key (str, optional): The key to look for.

    Returns:
        list: List of (location_path, value) tuples.
    """
    output_values = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == output_key:
                if isinstance(value, (int, float)):
                    # Handle single numeric value
                    location = current_path if current_path else "root"
                    output_values.append((location, value))
                elif isinstance(value, list):
                    # Handle list of numeric values
                    for i, item in enumerate(value):
                        if isinstance(item, (int, float)):
                            location = f"{current_path}.{key}[{i}]" if current_path else f"{key}[{i}]"
                            output_values.append((location, item))
            elif isinstance(value, (dict, list)):
                # Recurse into nested structures
                new_path = f"{current_path}.{key}" if current_path else key
                output_values.extend(find_single_input_values(value, new_path, output_key))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                output_values.extend(find_single_input_values(item, new_path, output_key))

    return output_values


def find_two_input_values(data, current_path="", input_key1=None, input_key2=None):
    """
    Recursively find all pairs of *source* values for two-input mode.
    Note: input_key2 can be a key name (str) or a literal value (int/float).

    Args:
        data (dict or list): The data structure to search.
        current_path (str, optional): Current path string.
        input_key1 (str, optional): First key to find.
        input_key2 (str or numeric, optional): Second key or value.

    Returns:
        list: List of (location_path, value1, value2) tuples.
    """
    if input_key1 is None:
        return []

    output_values = []
    is_two_keys = isinstance(input_key2, str)
    if isinstance(data, dict):
        # Determine the second input value, which might be a literal constant
        val2 = None
        if is_two_keys:
            # Two keys: check if the second key is present in THIS dictionary
            if input_key2 in data:
                val2 = data[input_key2]
        else:
            # One key, one literal value
            val2 = input_key2

        # Check if input_key1 is present
        if input_key1 in data and val2 is not None:
            val1 = data[input_key1]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Case 1: Single numeric values
                location = current_path if current_path else "root"
                output_values.append((location, val1, val2))
            elif isinstance(val1, list) and isinstance(val2, list) and len(val1) == len(val2):
                # Case 2: Lists of numeric values (only if input_key2 is also a list)
                for i in range(len(val1)):
                    item1 = val1[i]
                    item2 = val2[i]
                    if isinstance(item1, (int, float)) and isinstance(item2, (int, float)):
                        location = f"{current_path}[{i}]" if current_path else f"[{i}]"
                        output_values.append((location, item1, item2))

        # Recurse into nested structures
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                new_path = f"{current_path}.{key}" if current_path else key
                # Note: We pass the original input_key2 to maintain the structure/literal value
                output_values.extend(find_two_input_values(value, new_path, input_key1, input_key2))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                output_values.extend(find_two_input_values(item, new_path, input_key1, input_key2))

    return output_values


def process_pattern_files(
    root_directory,
    filename_pattern="log_*.json",
    output_key="km",
    process_func=None,
    update_val=0,
    input_keys=(None, None),
):
    """
    Search for all files starting with given filename pattern in subdirectories and modify values.

    Args:
        root_directory (str): Root directory to start search.
        filename_pattern (str, optional): Glob pattern for filenames.
        output_key (str, optional): Key to modify.
        process_func (callable, optional): Processing function.
        update_val (numeric, optional): Value for single-input updates.
        input_keys (tuple, optional): (key1, key2) for two-input mode.
    """
    assert process_func is not None, "Argument process_func must be provided"

    # Find all matching files
    pattern = os.path.join(root_directory, "**", filename_pattern)
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"No files matching '{filename_pattern}' found in {root_directory}")
        return

    # Set parameters for presence of input_keys
    if len(input_keys) >= 1 and input_keys[0] is not None:
        has_2_inputs = True
        if len(input_keys) < 2 or input_keys[1] is None:
            key_value = update_val
        else:
            key_value = input_keys[1]
    else:
        has_2_inputs = False

    print(f"Found {len(files)} files to process:")
    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Track if any modifications were made and process based on the structure type
            modified = False
            if isinstance(data, dict):
                # Case 1: dict of dicts
                if has_2_inputs:
                    modified = process_dict_two_inputs(data, input_keys[0], key_value, output_key, process_func)
                else:
                    modified = process_dict_of_dicts(data, output_key, process_func, update_val)
            elif isinstance(data, list):
                # Case 2: list of dicts of dicts
                if has_2_inputs:
                    modified = process_list_two_inputs(data, input_keys[0], key_value, output_key, process_func)
                else:
                    modified = process_list_of_dicts(data, output_key, process_func, update_val)
            else:
                print(f"- Warning: Unexpected data structure in {file_path}")
                continue

            # Write back to file if modifications were made
            if modified:
                with open(file_path, "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=2)
                print(f"- Successfully updated '{file_path}'")
            else:
                print(f"- Warning: could not find '{output_key}' field(s) to modify in '{file_path}'")
        except json.JSONDecodeError as e:
            print(f"- ERROR {e}: could not read contents from '{file_path}'")
        except Exception as e:
            print(f"- ERROR {e}: could not process '{file_path}'")


def process_file(file_path, output_key="km", process_func=None, update_val=0, input_keys=(None, None)):
    """
    Modify a single file by path.

    Args:
        file_path (str): Path to the file.
        output_key (str, optional): Key to modify.
        process_func (callable, optional): function.
        update_val (numeric, optional): update value.
        input_keys (tuple, optional): (key1, key2).

    Returns:
        bool: True if successful, False otherwise.
    """
    assert process_func is not None, "Argument process_func must be provided"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False

    if not os.path.isfile(file_path):
        print(f"Path is not a file: {file_path}")
        return False

    # Set parameters for presence of input_keys
    if len(input_keys) >= 1 and input_keys[0] is not None:
        has_2_inputs = True
        if len(input_keys) < 2 or input_keys[1] is None:
            key_value = update_val
        else:
            key_value = input_keys[1]
    else:
        has_2_inputs = False

    print(f"Processing single file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Track if any modifications were made and process based on the structure type
        modified = False
        if isinstance(data, dict):
            # Case 1: dict of dicts
            if has_2_inputs:
                modified = process_dict_two_inputs(data, input_keys[0], key_value, output_key, process_func)
            else:
                modified = process_dict_of_dicts(data, output_key, process_func, update_val)
        elif isinstance(data, list):
            # Case 2: list of dicts of dicts
            if has_2_inputs:
                modified = process_list_two_inputs(data, input_keys[0], key_value, output_key, process_func)
            else:
                modified = process_list_of_dicts(data, output_key, process_func, update_val)
        else:
            print(f"- Warning: Unexpected data structure in {file_path}")
            return False

        # Write back to file if modifications were made
        if modified:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
            print(f"- Successfully updated '{file_path}'")
            return True
        else:
            print(f"- Warning: could not find '{output_key}' field(s) to modify in '{file_path}'")
            return False
    except json.JSONDecodeError as e:
        print(f"- ERROR {e}: could not read contents from '{file_path}'")
        return False
    except Exception as e:
        print(f"- ERROR {e}: could not process '{file_path}'")
        return False


def process_pattern_files_statistics(
    root_directory,
    filename_pattern="log_*.json",
    output_filename="output.json",
    output_key="km",
    process_func=None,
):
    """
    Search for all files starting with given filename pattern in subdirectories,
    process values, and write output to new files with specified filename in same directories.

    Args:
        root_directory (str): Search root.
        filename_pattern (str, optional): Input file pattern.
        output_filename (str, optional): Name of the new output file.
        output_key (str, optional): Key to read/process.
        process_func (callable, optional): Aggregation/Processing function.

    Returns:
        int: Number of files processed.
    """
    assert process_func is not None, "Argument process_func must be provided"
    assert output_filename is not None, "Output filename must be provided"

    # Find all matching files
    pattern = os.path.join(root_directory, "**", filename_pattern)
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"No files matching '{filename_pattern}' found in {root_directory}")
        return

    print(f"Found {len(files)} files to process:")
    processed_count = 0
    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Track if any modifications were made and process based on the structure type
            key_values_found = find_single_input_values(data, output_key=output_key)
            grouped_values = {}
            if key_values_found:
                # Group values by field name (the part after the last dot)
                for location, value in key_values_found:
                    # Extract field name from location (e.g., '[0].hexaly_vrpp0.84_gamma1' -> 'hexaly_vrpp0.84_gamma1')
                    field_name = location.split(".", 1)[-1] if "." in location else location

                    if field_name not in grouped_values:
                        grouped_values[field_name] = []
                    grouped_values[field_name].append(value)

            # Write to output file if modifications were made
            if grouped_values:
                # Get directory of current file and create output path
                file_dir = os.path.dirname(file_path)
                output_path = os.path.join(file_dir, output_filename)

                # Write processed data to output file
                if os.path.exists(output_path):
                    with open(output_path, "r", encoding="utf-8") as file:
                        processed_data = json.load(file)
                else:
                    processed_data = {}

                for field_name, values in grouped_values.items():
                    if values:
                        if field_name not in processed_data:
                            processed_data[field_name] = {}
                        processed_data[field_name][output_key] = process_func(values)

                with open(output_path, "w", encoding="utf-8") as output_file:
                    json.dump(processed_data, output_file, indent=2)

                print(f"- Successfully wrote output to '{output_path}'")
                processed_count += 1
            else:
                print(f"- Warning: could not find '{output_key}' field(s) to process in '{file_path}'")
        except json.JSONDecodeError as e:
            print(f"- ERROR {e}: could not read contents from '{file_path}'")
        except Exception as e:
            print(f"- ERROR {e}: could not process '{file_path}'")

    print(f"\nProcessing completed: {processed_count}/{len(files)} files processed successfully")
    return processed_count


def process_file_statistics(file_path, output_filename="output.json", output_key="km", process_func=None):
    """
    Process a single file and write output to a new file with specified filename in the same directory.

    Args:
        file_path (str): Input file path.
        output_filename (str, optional): Output filename.
        output_key (str, optional): Key to process.
        process_func (callable, optional): Processing function.

    Returns:
        bool: True if successful, False otherwise.
    """
    assert process_func is not None, "Argument process_func must be provided"
    assert output_filename is not None, "Output filename must be provided"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False

    if not os.path.isfile(file_path):
        print(f"Path is not a file: {file_path}")
        return False

    print(f"Processing single file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Create a copy of the data to modify (preserve original)
        processed_data = json.loads(json.dumps(data))  # Deep copy

        # Track if any modifications were made and process based on the structure type
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

        # Write to output file if modifications were made
        if grouped_values:
            # Get directory of current file and create output path
            file_dir = os.path.dirname(file_path)
            output_path = os.path.join(file_dir, output_filename)

            # Write processed data to output file
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as file:
                    processed_data = json.load(file)
            else:
                processed_data = {}

            for field_name, values in grouped_values.items():
                if values:
                    if field_name not in processed_data:
                        processed_data[field_name] = {}
                    processed_data[field_name][output_key] = process_func(values)

            with open(output_path, "w", encoding="utf-8") as output_file:
                json.dump(processed_data, output_file, indent=2)

            print(f"- Successfully wrote output to '{output_path}'")
            return True
        else:
            print(f"- Warning: could not find '{output_key}' field(s) to process in '{file_path}'")
            return False
    except json.JSONDecodeError as e:
        print(f"- ERROR {e}: could not read contents from '{file_path}'")
        return False
    except Exception as e:
        print(f"- ERROR {e}: could not process '{file_path}'")
        return False
