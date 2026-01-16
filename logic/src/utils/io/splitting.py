"""
File splitting and reassembly utilities.
"""
import os
import shutil
import zipfile

import pandas as pd


def split_file(file_path, max_part_size, output_dir):
    """
    Split a large CSV or Excel file into smaller chunks based on size.

    Args:
        file_path (str): Path to the large file.
        max_part_size (int): Maximum size of each chunk in bytes.
        output_dir (str): Directory to save the chunked files.

    Returns:
        list: List of paths to the chunked files.
    """
    file_name, file_ext = os.path.splitext(os.path.basename(file_path))
    assert file_ext in [".csv", ".xls", ".xlsx"]

    chunk_files = []
    chunk_number = 1
    if file_ext == ".csv":
        df = pd.read_csv(file_path)

        def write_chunk(df, file):
            """Writes chunk to CSV."""
            df.to_csv(file, index=False)

    else:
        df = pd.read_excel(file_path)

        def write_chunk(df, file):
            """Writes chunk to Excel."""
            df.to_excel(file, index=False)

    # Split the file's DataFrame into chunks
    num_rows = df.shape[0]
    # Check if file is small enough (memory usage calculation approx)
    mem_usage = df.memory_usage(deep=True).sum()
    if mem_usage <= max_part_size:
        # No split needed, just copy
        out_file = os.path.join(output_dir, os.path.basename(file_path))
        write_chunk(df, out_file)
        return [out_file]

    rows_per_chunk = int(max_part_size // (mem_usage // num_rows))
    if rows_per_chunk < 1:
        rows_per_chunk = 1

    # Calculate number of digits needed for zero padding
    num_chunks = (num_rows + rows_per_chunk - 1) // rows_per_chunk
    pad_len = len(str(num_chunks))

    for i in range(0, num_rows, rows_per_chunk):
        chunk = df.iloc[i : i + rows_per_chunk]
        chunk_file = os.path.join(output_dir, f"{file_name}_part{str(chunk_number).zfill(pad_len)}{file_ext}")
        write_chunk(chunk, chunk_file)
        chunk_files.append(chunk_file)
        chunk_number += 1
    return chunk_files


def chunk_zip_content(zip_path, max_part_size, data_dir):
    """
    Extract a ZIP archive and chunk files larger than max_part_size into multiple parts.

    Args:
        zip_path (str): Path to the original ZIP archive.
        max_part_size (int): Maximum size of each part in bytes.
        data_dir (str): Directory to save the chunked files to.

    Returns:
        list: List of chunked file paths.

    Raises:
        Exception: If directory creation fails.
    """
    # Create a temporary directory and extract the ZIP contents
    temp_dir = os.path.join(data_dir, "temp_extracted")
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except Exception:
        raise Exception("directories to save zip files do not exist and could not be created")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
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

    Args:
        data_dir (str): Directory of the chunked files and where to save the reassembled files to.

    Returns:
        list: List of reassembled file names.
    """
    file_ext = None
    filename = None
    data_files = []

    if not os.path.exists(data_dir):
        return []

    dir_files = sorted(os.listdir(data_dir))
    dfs = []
    for file_id, file in enumerate(dir_files):
        # Check if we moved to a new file group
        if filename is not None and filename not in file:
            if dfs:
                comb_df = pd.concat(dfs, ignore_index=True)
                out_path = os.path.join(data_dir, "{}{}".format(filename, file_ext))
                (
                    comb_df.to_csv(out_path, index=False)
                    if file_ext == ".csv"
                    else comb_df.to_excel(out_path, index=False)
                )
                data_files.append(os.path.basename(out_path))
                print(f"File reassembled and saved to: {out_path}")
            filename, file_ext = None, None
            dfs = []

        if "_part" in file:
            base_name, actual_ext = os.path.splitext(file)
            if "_part" in base_name:
                filename_without_part = base_name.rsplit("_part", 1)[0]

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
                if file_ext == ".csv":
                    df = pd.read_csv(chunk_path)
                else:
                    # Specify engine for Excel files
                    try:
                        df = pd.read_excel(chunk_path, engine="openpyxl")
                    except ImportError:
                        try:
                            df = pd.read_excel(chunk_path, engine="xlrd")
                        except ImportError:
                            df = pd.read_excel(chunk_path)
                dfs.append(df)

    # Flush the last group
    if dfs and filename:
        comb_df = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(data_dir, "{}{}".format(filename, file_ext))
        (comb_df.to_csv(out_path, index=False) if file_ext == ".csv" else comb_df.to_excel(out_path, index=False))
        data_files.append(os.path.basename(out_path))
        print(f"File reassembled and saved to: {out_path}")

    return data_files
