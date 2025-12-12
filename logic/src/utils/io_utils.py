import os
import json
import glob
import shutil
import signal
import zipfile
import pandas as pd
import logic.src.utils.definitions as udef

from collections.abc import Iterable


def read_output(json_path, policies, lock=None):
    if lock is not None: lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        with open(json_path) as json_file:
            json_data = json.load(json_file)
    finally:
        if lock is not None: lock.release()
    tmp = []
    for key, val in json_data.items():
        if key in policies:
            tmp.append(val)
    tmp = pd.DataFrame(tmp).values.transpose().tolist()
    return tmp


def read_json(json_path, lock=None):
    if lock is not None: lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        with open(json_path) as json_file:
            json_data = json.load(json_file)
    finally:
        if lock is not None: lock.release()
    return json_data


def zip_directory(input_dir: str, output_zip: str):
    """
    Create a zip archive of a directory.
    :param input_dir: Path to the directory to zip.
    :param output_zip: Path to save the zip archive to.
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to the zip archive, preserving the directory structure
                arcname = os.path.relpath(file_path, input_dir)
                zipf.write(file_path, arcname)
    print(f"Directory '{input_dir}' zipped to '{output_zip}'")
    return


def extract_zip(input_zip: str, output_dir: str):
    with zipfile.ZipFile(input_zip, 'r') as zipf:
        zipf.extractall(output_dir)
    print(f"Zip archive '{input_zip}' was extracted to directory '{output_dir}'")
    return


def split_file(file_path, max_part_size, output_dir):
    """
    Split a large CSV|Excel file into smaller chunks.
    :param file_path: Path to the large file.
    :param max_part_size: Maximum size of each chunk in bytes.
    :param output_dir: Directory to save the chunked files.
    :return: List of paths to the chunked files.
    """
    file_name, file_ext = os.path.splitext(os.path.basename(file_path))
    assert file_ext in ['.csv', '.xls', '.xlsx']

    chunk_files = []
    chunk_number = 1
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
        write_chunk = lambda df, file: df.to_csv(file, index=False)
    else:
        df = pd.read_excel(file_path)
        write_chunk = lambda df, file: df.to_excel(file, index=False)

    # Split the file's DataFrame into chunks 
    num_rows = df.shape[0]
    rows_per_chunk = int(max_part_size // (df.memory_usage(deep=True).sum() // num_rows))
    for i in range(0, num_rows, rows_per_chunk):
        chunk = df.iloc[i:i + rows_per_chunk]
        chunk_file = os.path.join(output_dir, f"{file_name}_part{chunk_number}{file_ext}")
        write_chunk(chunk, chunk_file)
        chunk_files.append(chunk_file)
        chunk_number += 1
    return chunk_files


def chunk_zip_content(zip_path, max_part_size, data_dir):
    """
    Extract a ZIP archive and chunk files larger than max_part_size into multiple parts.
    :param zip_path: Path to the original ZIP archive.
    :param max_part_size: Maximum size of each part in bytes.
    :param data_dir: Directory to save the chunked files to.
    :return: List of chunked file paths.
    """
    # Create a temporary directory and extract the ZIP contents
    temp_dir = os.path.join(data_dir, "temp_extracted")
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except Exception:
        raise Exception("directories to save zip files do not exist and could not be created")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    all_files = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    # Split large files into chunks, otherwise copy directly to data directory
    final_files = []
    for file in all_files:
        file_size = os.path.getsize(file)
        if file_size > max_part_size:
            print(f"Splitting large CSV file: {file} ({file_size / (1000**2):.2f} MB)")
            out_files = split_file(file, max_part_size, data_dir)
            final_files.extend(out_files)
        else:
            out_file = shutil.copy(file, data_dir)
            final_files.append(out_file)
    shutil.rmtree(temp_dir)
    return final_files


def reassemble_files(data_dir):
    """
    Reassemble several files from their chunks.
    :param data_dir: Directory of the chunked files and where to save the reassembled files to.
    :return: List of reassembled file names.
    """
    file_ext = None
    filename = None
    data_files = []
    dir_files = sorted(os.listdir(data_dir))
    num_files = len(dir_files)
    for file_id, file in enumerate(dir_files):
        if filename is not None and (filename not in file or file_id == num_files - 1):
            comb_df = pd.concat(dfs, ignore_index=True)
            out_path = os.path.join(data_dir, "{}{}".format(filename, file_ext))
            comb_df.to_csv(out_path, index=False) if file_ext == '.csv' else comb_df.to_excel(out_path, index=False)
            data_files.append(os.path.basename(out_path))
            filename, file_ext = None, None
            print(f"File reassembled and saved to: {out_path}")
            
        if "_part" in file:
            base_name, actual_ext = os.path.splitext(file)
            if "_part" in base_name:
                filename_without_part = base_name.rsplit('_part', 1)[0]
                
                # Skip if the original file already exists
                original_filename = f"{filename_without_part}{actual_ext}"
                if original_filename in dir_files: 
                    continue
                    
                if filename is None:
                    dfs = []
                    filename = filename_without_part
                    file_ext = actual_ext
            
            if filename and filename in file:
                chunk_path = os.path.join(data_dir, file)
                if file_ext == '.csv':
                    df = pd.read_csv(chunk_path)
                else:
                    # Specify engine for Excel files
                    try:
                        df = pd.read_excel(chunk_path, engine='openpyxl')
                    except ImportError:
                        try:
                            df = pd.read_excel(chunk_path, engine='xlrd')
                        except ImportError:
                            df = pd.read_excel(chunk_path)
                dfs.append(df)      
    return data_files


def process_dict_of_dicts(data_dict, output_key='km', process_func=None, update_val=0):
    """
    Process a dictionary of dictionaries and apply process_func to key values.
    Supports single numeric values OR lists of numeric values.
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


def process_list_of_dicts(data_list, output_key='km', process_func=None, update_val=0):
    """Process a list of dictionaries of dictionaries"""
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
    Process a dictionary of dictionaries and apply process_func using two inputs:
    [output_key] = process_func([input_key1], [input_key2_or_val])
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
                    if process_func.__name__ == '/' and input_val2 == 0:
                        print(f"   -- WARNING: Skipping division by zero in '{key}' (Input 2 is zero)")
                        continue
                        
                    new_val = process_func(input_val1, input_val2)
                    value_dict[output_key] = new_val
                    modified = True
                    input2_name = input_key2_or_val if is_two_keys else 'value'
                    print(f"   -- Calculated '{output_key}' in '{key}' ({input_key1} {process_func.__name__} {input2_name}): {new_val}")
                elif isinstance(input_val1, list) and isinstance(input_val2, list) and len(input_val1) == len(input_val2):
                    # Check for list of numeric values for both inputs
                    new_list = []
                    list_modified = False
                    input2_name = input_key2_or_val if is_two_keys else 'value'
                    for i in range(len(input_val1)):
                        item1 = input_val1[i]
                        item2 = input_val2[i]
                        if isinstance(item1, (int, float)) and isinstance(item2, (int, float)): 
                            if process_func.__name__ == '/' and item2 == 0:
                                print(f"   -- WARNING: Skipping division by zero for list item {i} in '{key}'")
                                # Try to keep the original output value if it exists, otherwise use None
                                new_list.append(value_dict.get(output_key, [])[i] if output_key in value_dict and i < len(value_dict[output_key]) else None)
                                continue

                            new_item = process_func(item1, item2)
                            new_list.append(new_item)
                            list_modified = True
                            print(f"   -- Calculated list item {i} of '{output_key}' in '{key}' ({input_key1}[{i}] {process_func.__name__} {input2_name}[{i}]): {new_item}")
                        else:
                            # If non-numeric, try to keep the original output value if it exists
                            new_list.append(value_dict.get(output_key, [])[i] if output_key in value_dict and i < len(value_dict[output_key]) else None)

                    if list_modified:
                        value_dict[output_key] = new_list
                        modified = True
                else:
                    print(f"   -- Warning: Input values for '{input_key1}' and '{input_key2_or_val}' are not compatible numeric types or lists in '{key}'")
    return modified


def process_list_two_inputs(data_list, input_key1, input_key2_or_val, output_key, process_func):
    """Process a list of dictionaries of dictionaries using two inputs"""
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
    Returns list of (location_path, value) tuples.
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
    Returns list of (location_path, value1, value2) tuples.
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


def process_pattern_files(root_directory, filename_pattern="log_*.json", output_key='km', process_func=None, update_val=0, input_keys=(None, None)):
    """
    Search for all files starting with given filename pattern in subdirectories and modify values
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
            with open(file_path, 'r', encoding='utf-8') as file:
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
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=2)
                print(f"- Successfully updated '{file_path}'")
            else:
                print(f"- Warning: could not find '{output_key}' field(s) to modify in '{file_path}'")
        except json.JSONDecodeError as e:
            print(f"- ERROR {e}: could not read contents from '{file_path}'")
        except Exception as e:
            print(f"- ERROR {e}: could not process '{file_path}'")


def preview_changes(root_directory, output_key='km', filename_pattern="log_*.json", process_func=None, update_val=0, input_keys=(None, None)):
    """
    Preview what changes will be made without actually modifying files
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
        key_value = update_val # Second input is the literal update_val
        input2_name = str(update_val)
    else:
        has_2_inputs = False
    
    print(f"Preview mode - found {len(pattern_files)} files:")
    for file_path in pattern_files:
        print(f"\nFile: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            if has_2_inputs:
                # Two-input mode: find pairs of keys (input_key2 can be a string key or a value)
                key_values_found = find_two_input_values(data, input_key1=key1, input_key2=key_value)
                if key_values_found:
                    for location, value1, value2 in key_values_found:
                        new_value = process_func(value1, value2)
                        print(f"- Would calculate and write to key '{output_key}' at '{location}': {value1} {process_func.__name__} {input2_name} → {new_value}")
                else:
                    print(f"- No suitable input pairs ({key1}, {input2_name}) found")
            else:
                # Single-input mode: find the output key
                key_values_found = find_single_input_values(data, output_key=output_key)
                if key_values_found:
                    for location, old_value in key_values_found:
                        new_value = process_func(old_value, update_val)
                        print(f"- Would update file '{location}' and key '{output_key}': {old_value} {process_func.__name__} {update_val} → {new_value}")
                else:
                    print(f"- No '{output_key}' values found")
        except Exception as e:
            print(f"- ERROR {e}: could not read '{file_path}'")


def process_file(file_path, output_key='km', process_func=None, update_val=0, input_keys=(None, None)):
    """
    Modify a single file by path
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
        with open(file_path, 'r', encoding='utf-8') as file:
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
            with open(file_path, 'w', encoding='utf-8') as file:
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


def preview_file_changes(file_path, output_key='km', process_func=None, update_val=0, input_keys=(None, None)):
    """
    Preview changes for a single file without modifying it
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
        key_value = update_val # Second input is the literal update_val
        input2_name = str(update_val)
    else:
        has_2_inputs = False

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if has_2_inputs:
            # Two-input mode: find pairs of keys
            key_values_found = find_two_input_values(data, input_key1=key1, input_key2=key_value)
            if key_values_found:
                for location, value1, value2 in key_values_found:
                    new_value = process_func(value1, value2)
                    print(f"- Would calculate and write to key '{output_key}' at '{location}': {value1} {process_func.__name__} {input2_name} → {new_value}")
            else:
                print(f"- No suitable input pairs ({key1}, {input2_name}) found")
        else:
            # Single-input mode: find the output key
            key_values_found = find_single_input_values(data, output_key=output_key)
            if key_values_found:
                for location, old_value in key_values_found:
                    new_value = process_func(old_value, update_val)
                    print(f"- Would update file '{location}' and key '{output_key}': {old_value} {process_func.__name__} {update_val} → {new_value}")
            else:
                print(f"- No '{output_key}' values found")
    except Exception as e:
        print(f"- ERROR {e}: could not read '{file_path}'")


def process_pattern_files_statistics(root_directory, filename_pattern="log_*.json", output_filename="output.json", output_key='km', process_func=None):
    """
    Search for all files starting with given filename pattern in subdirectories, 
    process values, and write output to new files with specified filename in same directories.
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
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Track if any modifications were made and process based on the structure type
            key_values_found = find_single_input_values(data, output_key=output_key)
            if key_values_found:
                # Group values by field name (the part after the last dot)
                grouped_values = {}
                for location, value in key_values_found:
                    # Extract field name from location (e.g., '[0].hexaly_vrpp0.84_gamma1' -> 'hexaly_vrpp0.84_gamma1')
                    field_name = location.split('.', 1)[-1] if '.' in location else location
                    
                    if field_name not in grouped_values:
                        grouped_values[field_name] = []
                    grouped_values[field_name].append(value)

            # Write to output file if modifications were made
            if grouped_values:
                # Get directory of current file and create output path
                file_dir = os.path.dirname(file_path)
                output_path = os.path.join(file_dir, output_filename)
                
                # Write processed data to output file
                with open(output_path, 'r', encoding='utf-8') as file:
                    processed_data = json.load(file)

                for field_name, values in grouped_values.items():
                    if values:
                        processed_data[field_name][output_key] = process_func(values)

                with open(output_path, 'w', encoding='utf-8') as output_file:
                    json.dump(processed_data, output_file, indent=2)
                
                print(f"- Successfully wrote output to '{output_path}'")
                processed_count += 1
            else:
                print(f"- Warning: could not find '{output_key}' field(s) to process in '{file_path}'")     
        except json.JSONDecodeError as e:
            print(f"- ERROR {e}: could not read contents from '{file_path}'")
        except Exception as e:
            print("Last field:", field_name)
            print(f"- ERROR {e}: could not process '{file_path}'")
    
    print(f"\nProcessing completed: {processed_count}/{len(files)} files processed successfully")
    return processed_count


def preview_pattern_files_statistics(root_directory, filename_pattern="log_*.json", output_filename="output.json", output_key='km', process_func=None):
    """
    Preview changes for pattern files statistics operation without modifying files
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
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Track if any modifications would be made
            key_values_found = find_single_input_values(data, output_key=output_key)
            if key_values_found:
                # Group values by field name (the part after the last dot)
                grouped_values = {}
                for location, value in key_values_found:
                    # Extract field name from location (e.g., '[0].hexaly_vrpp0.84_gamma1' -> 'hexaly_vrpp0.84_gamma1')
                    field_name = location.split('.', 1)[-1] if '.' in location else location
                    
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
                        print(f"- Would write to '{output_path}': '{field_name}.{output_key}' = {process_func.__name__}({values}) = {new_value}")
                
                files_with_changes += 1
            else:
                print(f"- No '{output_key}' values found that would be processed")
        except Exception as e:
            print(f"- ERROR {e}: could not read '{file_path}'")
    print(f"\nSummary: {files_with_changes}/{len(files)} files would be processed")


def process_file_statistics(file_path, output_filename="output.json", output_key='km', process_func=None):
    """
    Process a single file and write output to a new file with specified filename in the same directory.
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
        with open(file_path, 'r', encoding='utf-8') as file:
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
                field_name = location.split('.', 1)[-1] if '.' in location else location
                
                if field_name not in grouped_values:
                    grouped_values[field_name] = []
                grouped_values[field_name].append(value)
        
        # Write to output file if modifications were made
        if grouped_values:
            # Get directory of current file and create output path
            file_dir = os.path.dirname(file_path)
            output_path = os.path.join(file_dir, output_filename)
            
            # Write processed data to output file
            with open(output_path, 'r', encoding='utf-8') as file:
                processed_data = json.load(file)

            for field_name, values in grouped_values.items():
                if values:
                    processed_data[field_name][output_key] = process_func(values)

            with open(output_path, 'w', encoding='utf-8') as output_file:
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


def preview_file_statistics(file_path, output_filename="output.json", output_key='km', process_func=None):
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
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Track if any modifications would be made
        key_values_found = find_single_input_values(data, output_key=output_key)
        if key_values_found:
            # Group values by field name (the part after the last dot)
            grouped_values = {}
            for location, value in key_values_found:
                # Extract field name from location (e.g., '[0].hexaly_vrpp0.84_gamma1' -> 'hexaly_vrpp0.84_gamma1')
                field_name = location.split('.', 1)[-1] if '.' in location else location
                
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
                    print(f"- Would write to '{output_path}': '{field_name}.{output_key}' = {process_func.__name__}({values}) = {new_value}")
            
            return True
        else:
            print(f"- No '{output_key}' values found that would be processed")
            return False
    except Exception as e:
        print(f"- ERROR {e}: could not read '{file_path}'")


def confirm_proceed(default_no=True, operation_name='update'):
    """
    Ask user for confirmation with timeout and default response
    Returns True if user confirms, False if user cancels or timeout
    """
    def timeout_handler(signum, frame):
        raise TimeoutError()
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(udef.CONFIRM_TIMEOUT)
    try:
        prompt = f"\nProceed with {operation_name}? (y/n) [{'n' if default_no else 'y'}] (timeout {udef.CONFIRM_TIMEOUT}s): "
        response = input(prompt).strip().lower()
        signal.alarm(0)  # Cancel timeout
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            # Default response if user just presses enter
            return not default_no    
    except TimeoutError:
        print(f"\nTimeout reached. Defaulting to {'no' if default_no else 'yes'}.")
        return not default_no
    except KeyboardInterrupt:
        print(f"\nOperation interrupted by user. Defaulting to {'no' if default_no else 'yes'}.")
        return not default_no


def compose_dirpath(fun):
    def inner(home_dir, ndays, nbins, output_dir, area, *args, **kwargs):
        if not isinstance(nbins, Iterable):
            dir_path = os.path.join(home_dir, "assets", output_dir, f"{ndays}_days", f"{area}_{nbins}")
            return fun(dir_path, *args, **kwargs)

        dir_paths = []
        for gs in nbins:
            dir_paths.append(os.path.join(home_dir, "assets", output_dir, f"{ndays}_days", f"{area}_{gs}"))
        return fun(dir_paths, *args, **kwargs)
    return inner