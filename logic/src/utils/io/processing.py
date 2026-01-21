from __future__ import annotations

import glob
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast


def process_dict_of_dicts(
    data_dict: Dict[str, Any],
    output_key: str = "km",
    process_func: Optional[Callable[[Any, Any], Any]] = None,
    update_val: Union[int, float] = 0,
) -> bool:
    """
    Process a dictionary of dictionaries and apply process_func to key values.
    Supports single numeric values OR lists of numeric values.

    Args:
        data_dict: The dictionary to process.
        output_key: The key within inner dicts to modify. Defaults to 'km'.
        process_func: function(old_val, update_val) -> new_val.
        update_val: The second argument for process_func. Defaults to 0.

    Returns:
        True if any modification was made, False otherwise.
    """
    assert process_func is not None, "Argument process_func must be provided"

    modified: bool = False
    for key, value in data_dict.items():
        if isinstance(value, dict) and output_key in value:
            target_value: Any = value[output_key]

            if isinstance(target_value, (int, float)):
                modified = True
                old_val: Union[int, float] = target_value
                value[output_key] = process_func(old_val, update_val)
                print(f"   -- Updated '{output_key}' in '{key}': {old_val} → {value[output_key]}")

            elif isinstance(target_value, list):
                list_modified: bool = False
                for i, item in enumerate(target_value):
                    if isinstance(item, (int, float)):
                        list_modified = True
                        old_val = item
                        target_value[i] = process_func(old_val, update_val)
                        print(f"   -- Updated '{output_key}[{i}]' in '{key}': {old_val} → {target_value[i]}")

                if list_modified:
                    modified = True

            else:
                print(f"   -- Warning: '{output_key}' value is not numeric or a list in '{key}'")

    return modified


def process_list_of_dicts(
    data_list: List[Dict[str, Any]],
    output_key: str = "km",
    process_func: Optional[Callable[[Any, Any], Any]] = None,
    update_val: Union[int, float] = 0,
) -> bool:
    """
    Process a list of dictionaries of dictionaries.

    Args:
        data_list: The list of dictionaries to process.
        output_key: The key to modify. Defaults to 'km'.
        process_func: The function to apply.
        update_val: The update value. Defaults to 0.

    Returns:
        True if any modification was made.
    """
    assert process_func is not None, "Argument process_func must be provided"

    modified: bool = False
    for item in data_list:
        if isinstance(item, dict):
            if process_dict_of_dicts(item, output_key, process_func, update_val):
                modified = True
    return modified


def process_dict_two_inputs(
    data_dict: Dict[str, Any],
    input_key1: str,
    input_key2_or_val: Union[str, int, float],
    output_key: str,
    process_func: Callable[[Any, Any], Any],
) -> bool:
    """
    Process a dictionary of dictionaries and apply process_func using two inputs.
    Format: [output_key] = process_func([input_key1], [input_key2_or_val])

    Args:
        data_dict: The dictionary to process.
        input_key1: key for the first input operand.
        input_key2_or_val: key for second input, or a literal value.
        output_key: key to store the result.
        process_func: function(val1, val2) -> new_val.

    Returns:
        True if any modification was made.
    """
    assert process_func is not None, "Argument process_func must be provided"

    modified: bool = False
    is_two_keys: bool = isinstance(input_key2_or_val, str)
    for key, value_dict in data_dict.items():
        if isinstance(value_dict, dict) and input_key1 in value_dict:
            input_val1: Any = value_dict[input_key1]
            input_val2: Any = None

            if is_two_keys:
                if isinstance(input_key2_or_val, str) and input_key2_or_val in value_dict:
                    input_val2 = value_dict[input_key2_or_val]
            else:
                input_val2 = input_key2_or_val

            if input_val2 is not None:
                if isinstance(input_val1, (int, float)) and isinstance(input_val2, (int, float)):
                    if getattr(process_func, "__name__", "") == "/" and input_val2 == 0:
                        print(f"   -- WARNING: Skipping division by zero in '{key}' (Input 2 is zero)")
                        continue

                    new_val: Union[int, float] = process_func(input_val1, input_val2)
                    value_dict[output_key] = new_val
                    modified = True
                    input2_name: str = str(input_key2_or_val) if is_two_keys else "value"
                    print(
                        f"   -- Calculated '{output_key}' in '{key}' "
                        f"({input_key1} {getattr(process_func, '__name__', 'func')} {input2_name}): {new_val}"
                    )
                elif (
                    isinstance(input_val1, list) and isinstance(input_val2, list) and len(input_val1) == len(input_val2)
                ):
                    new_list: List[Any] = []
                    list_modified: bool = False
                    input2_name = str(input_key2_or_val) if is_two_keys else "value"
                    for i in range(len(input_val1)):
                        item1: Any = input_val1[i]
                        item2: Any = input_val2[i]
                        if isinstance(item1, (int, float)) and isinstance(item2, (int, float)):
                            if getattr(process_func, "__name__", "") == "/" and item2 == 0:
                                print(f"   -- WARNING: Skipping division by zero for list item {i} in '{key}'")
                                new_list.append(
                                    value_dict.get(output_key, [])[i]
                                    if output_key in value_dict and i < len(value_dict[output_key])
                                    else None
                                )
                                continue

                            new_item: Union[int, float] = process_func(item1, item2)
                            new_list.append(new_item)
                            list_modified = True
                        else:
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


def process_list_two_inputs(
    data_list: List[Dict[str, Any]],
    input_key1: str,
    input_key2_or_val: Union[str, int, float],
    output_key: str,
    process_func: Callable[[Any, Any], Any],
) -> bool:
    """
    Process a list of dictionaries of dictionaries using two inputs.

    Args:
        data_list: The list of dictionaries.
        input_key1: First input key.
        input_key2_or_val: Second input key or value.
        output_key: Output key.
        process_func: Processing function.

    Returns:
        True if modified.
    """
    assert process_func is not None, "Argument process_func must be provided"

    modified: bool = False
    for item in data_list:
        if isinstance(item, dict):
            if process_dict_two_inputs(item, input_key1, input_key2_or_val, output_key, process_func):
                modified = True
    return modified


def find_single_input_values(
    data: Union[Dict[str, Any], List[Any]], current_path: str = "", output_key: str = "km"
) -> List[Tuple[str, Union[int, float]]]:
    """
    Recursively find all *source* values for single-input mode.

    Args:
        data: The data structure to search.
        current_path: Current path in the recursive search.
        output_key: The key to look for.

    Returns:
        List of (location_path, value) tuples.
    """
    output_values: List[Tuple[str, Union[int, float]]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == output_key:
                if isinstance(value, (int, float)):
                    location: str = current_path if current_path else "root"
                    output_values.append((location, value))
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, (int, float)):
                            location = f"{current_path}.{key}[{i}]" if current_path else f"{key}[{i}]"
                            output_values.append((location, item))
            elif isinstance(value, (dict, list)):
                new_path: str = f"{current_path}.{key}" if current_path else key
                output_values.extend(find_single_input_values(value, new_path, output_key))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                output_values.extend(find_single_input_values(item, new_path, output_key))

    return output_values


def find_two_input_values(
    data: Union[Dict[str, Any], List[Any]],
    current_path: str = "",
    input_key1: Optional[str] = None,
    input_key2: Union[str, int, float, None] = None,
) -> List[Tuple[str, Union[int, float], Union[int, float]]]:
    """
    Recursively find all pairs of *source* values for two-input mode.

    Args:
        data: The data structure to search.
        current_path: Current path string.
        input_key1: First key to find.
        input_key2: Second key or value.

    Returns:
        List of (location_path, value1, value2) tuples.
    """
    if input_key1 is None:
        return []

    output_values: List[Tuple[str, Union[int, float], Union[int, float]]] = []
    is_two_keys: bool = isinstance(input_key2, str)
    if isinstance(data, dict):
        val2: Any = None
        if is_two_keys:
            if isinstance(input_key2, str) and input_key2 in data:
                val2 = data[input_key2]
        else:
            val2 = input_key2

        if input_key1 in data and val2 is not None:
            val1: Any = data[input_key1]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                location: str = current_path if current_path else "root"
                output_values.append((location, val1, val2))
            elif isinstance(val1, list) and isinstance(val2, list) and len(val1) == len(val2):
                for i in range(len(val1)):
                    item1: Any = val1[i]
                    item2: Any = val2[i]
                    if isinstance(item1, (int, float)) and isinstance(item2, (int, float)):
                        location = f"{current_path}[{i}]" if current_path else f"[{i}]"
                        output_values.append((location, item1, item2))

        for key, value in data.items():
            if isinstance(value, (dict, list)):
                new_path: str = f"{current_path}.{key}" if current_path else key
                output_values.extend(find_two_input_values(value, new_path, input_key1, input_key2))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                output_values.extend(find_two_input_values(item, new_path, input_key1, input_key2))

    return output_values


def process_pattern_files(
    root_directory: str,
    filename_pattern: str = "log_*.json",
    output_key: str = "km",
    process_func: Optional[Callable[[Any, Any], Any]] = None,
    update_val: Union[int, float] = 0,
    input_keys: Tuple[Optional[str], Union[str, int, float, None]] = (None, None),
) -> None:
    """
    Search for all files starting with given filename pattern in subdirectories and modify values.

    Args:
        root_directory: Root directory to start search.
        filename_pattern: Glob pattern for filenames.
        output_key: Key to modify.
        process_func: Processing function.
        update_val: Value for single-input updates.
        input_keys: (key1, key2) for two-input mode.
    """
    assert process_func is not None, "Argument process_func must be provided"

    pattern: str = os.path.join(root_directory, "**", filename_pattern)
    files: List[str] = glob.glob(pattern, recursive=True)
    if not files:
        print(f"No files matching '{filename_pattern}' found in {root_directory}")
        return

    has_2_inputs: bool = False
    key_value: Union[str, int, float, None] = None
    if len(input_keys) >= 1 and input_keys[0] is not None:
        has_2_inputs = True
        if len(input_keys) < 2 or input_keys[1] is None:
            key_value = update_val
        else:
            key_value = input_keys[1]

    print(f"Found {len(files)} files to process:")
    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data: Any = json.load(file)

            modified: bool = False
            if isinstance(data, dict):
                if has_2_inputs:
                    modified = process_dict_two_inputs(
                        data,
                        cast(str, input_keys[0]),
                        cast(Union[str, int, float], key_value),
                        output_key,
                        process_func,
                    )
                else:
                    modified = process_dict_of_dicts(data, output_key, process_func, update_val)
            elif isinstance(data, list):
                if has_2_inputs:
                    modified = process_list_two_inputs(
                        data,
                        cast(str, input_keys[0]),
                        cast(Union[str, int, float], key_value),
                        output_key,
                        process_func,
                    )
                else:
                    modified = process_list_of_dicts(data, output_key, process_func, update_val)
            else:
                print(f"- Warning: Unexpected data structure in {file_path}")
                continue

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


def process_file(
    file_path: str,
    output_key: str = "km",
    process_func: Optional[Callable[[Any, Any], Any]] = None,
    update_val: Union[int, float] = 0,
    input_keys: Tuple[Optional[str], Union[str, int, float, None]] = (None, None),
) -> bool:
    """
    Modify a single file by path.

    Args:
        file_path: Path to the file.
        output_key: Key to modify.
        process_func: function.
        update_val: update value.
        input_keys: (key1, key2).

    Returns:
        True if successful, False otherwise.
    """
    assert process_func is not None, "Argument process_func must be provided"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False

    if not os.path.isfile(file_path):
        print(f"Path is not a file: {file_path}")
        return False

    has_2_inputs: bool = False
    key_value: Union[str, int, float, None] = None
    if len(input_keys) >= 1 and input_keys[0] is not None:
        has_2_inputs = True
        if len(input_keys) < 2 or input_keys[1] is None:
            key_value = update_val
        else:
            key_value = input_keys[1]

    print(f"Processing single file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data: Any = json.load(file)

        modified: bool = False
        if isinstance(data, dict):
            if has_2_inputs:
                modified = process_dict_two_inputs(
                    data, cast(str, input_keys[0]), cast(Union[str, int, float], key_value), output_key, process_func
                )
            else:
                modified = process_dict_of_dicts(data, output_key, process_func, update_val)
        elif isinstance(data, list):
            if has_2_inputs:
                modified = process_list_two_inputs(
                    data, cast(str, input_keys[0]), cast(Union[str, int, float], key_value), output_key, process_func
                )
            else:
                modified = process_list_of_dicts(data, output_key, process_func, update_val)
        else:
            return False

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
    root_directory: str,
    filename_pattern: str = "log_*.json",
    output_filename: str = "output.json",
    output_key: str = "km",
    process_func: Optional[Callable[[List[Union[int, float]]], Union[int, float]]] = None,
) -> Optional[int]:
    """
    Search for all files starting with given filename pattern in subdirectories,
    process values, and write output to new files.
    """
    assert process_func is not None, "Argument process_func must be provided"
    assert output_filename is not None, "Output filename must be provided"

    pattern: str = os.path.join(root_directory, "**", filename_pattern)
    files: List[str] = glob.glob(pattern, recursive=True)
    if not files:
        print(f"No files matching '{filename_pattern}' found in {root_directory}")
        return None

    print(f"Found {len(files)} files to process:")
    processed_count: int = 0
    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data: Any = json.load(file)

            key_values_found: List[Tuple[str, Union[int, float]]] = find_single_input_values(
                data, output_key=output_key
            )
            grouped_values: Dict[str, List[Union[int, float]]] = {}
            if key_values_found:
                for location, value in key_values_found:
                    field_name: str = location.split(".", 1)[-1] if "." in location else location
                    if field_name not in grouped_values:
                        grouped_values[field_name] = []
                    grouped_values[field_name].append(value)

            if grouped_values:
                file_dir: str = os.path.dirname(file_path)
                output_path: str = os.path.join(file_dir, output_filename)

                processed_data: Dict[str, Any] = {}
                if os.path.exists(output_path):
                    with open(output_path, "r", encoding="utf-8") as file:
                        processed_data = json.load(file)

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


def process_file_statistics(
    file_path: str,
    output_filename: str = "output.json",
    output_key: str = "km",
    process_func: Optional[Callable[[List[Union[int, float]]], Union[int, float]]] = None,
) -> bool:
    """
    Process a single file and write output to a new file.
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
            data: Any = json.load(file)

        key_values_found: List[Tuple[str, Union[int, float]]] = find_single_input_values(data, output_key=output_key)
        grouped_values: Dict[str, List[Union[int, float]]] = {}
        if key_values_found:
            for location, value in key_values_found:
                field_name: str = location.split(".", 1)[-1] if "." in location else location
                if field_name not in grouped_values:
                    grouped_values[field_name] = []
                grouped_values[field_name].append(value)

        if grouped_values:
            file_dir: str = os.path.dirname(file_path)
            output_path: str = os.path.join(file_dir, output_filename)

            processed_data: Dict[str, Any] = {}
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as file:
                    processed_data = json.load(file)

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
