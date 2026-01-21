"""
File operation utilities.
"""

import json
import os
import signal
import zipfile
from collections.abc import Iterable

import logic.src.utils.definitions as udef


def read_json(json_path, lock=None):
    """
    Reads a JSON file safely with an optional lock.

    Args:
        json_path (str): Path to the JSON file.
        lock (threading.Lock, optional): Thread lock for safe access.

    Returns:
        dict or list: The parsed JSON data.
    """
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
    finally:
        if lock is not None:
            lock.release()
    return json_data


def zip_directory(input_dir: str, output_zip: str):
    """
    Create a zip archive of a directory.

    Args:
        input_dir (str): Path to the directory to zip.
        output_zip (str): Path to save the zip archive to.
    """
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to the zip archive, preserving the directory structure
                arcname = os.path.relpath(file_path, input_dir)
                zipf.write(file_path, arcname)
    print(f"Directory '{input_dir}' zipped to '{output_zip}'")
    return


def extract_zip(input_zip: str, output_dir: str):
    """
    Extracts a zip archive to a directory.

    Args:
        input_zip (str): Path to the zip archive.
        output_dir (str): Directory where contents will be extracted.
    """
    with zipfile.ZipFile(input_zip, "r") as zipf:
        zipf.extractall(output_dir)
    print(f"Zip archive '{input_zip}' was extracted to directory '{output_dir}'")
    return


def confirm_proceed(default_no=True, operation_name="update"):
    """
    Ask user for confirmation with timeout and default response
    Returns True if user confirms, False if user cancels or timeout
    """

    def timeout_handler(signum, frame):
        """Raises TimeoutError on signal."""
        raise TimeoutError()

    # Set timeout
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(udef.CONFIRM_TIMEOUT)
        try:
            prompt = f"\nProceed with {operation_name}? (y/n) [{'n' if default_no else 'y'}] (timeout {udef.CONFIRM_TIMEOUT}s): "
            response = input(prompt).strip().lower()
            signal.alarm(0)  # Cancel timeout
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
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
    except AttributeError:
        # Windows compatibility or if signal is not fully supported
        prompt = f"\nProceed with {operation_name}? (y/n) [{'n' if default_no else 'y'}]: "
        response = input(prompt).strip().lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        return not default_no


def compose_dirpath(fun):
    """
    Decorator to compose directory paths based on home_dir, ndays, num_bins, etc.

    Args:
        fun (callable): Function to decorate.

    Returns:
        callable: Decorated function.
    """

    def inner(home_dir, ndays, nbins, output_dir, area, *args, **kwargs):
        """
        Inner wrapper function.

        Args:
            home_dir (str): Home directory.
            ndays (int): Number of days.
            nbins (int or list): Number of bins or list of them.
            output_dir (str): Output directory name.
            area (str): Area name.
        """
        if not isinstance(nbins, Iterable):
            dir_path = os.path.join(home_dir, "assets", output_dir, f"{ndays}_days", f"{area}_{nbins}")
            return fun(dir_path, *args, **kwargs)

        dir_paths = []
        for gs in nbins:
            dir_paths.append(os.path.join(home_dir, "assets", output_dir, f"{ndays}_days", f"{area}_{gs}"))
        return fun(dir_paths, *args, **kwargs)

    return inner
