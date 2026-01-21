from __future__ import annotations

import json
import os
import signal
import threading
import zipfile
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import logic.src.utils.definitions as udef

T = TypeVar("T", bound=Callable[..., Any])


def read_json(json_path: str, lock: Optional[threading.Lock] = None) -> Union[Dict[str, Any], List[Any]]:
    """
    Reads a JSON file safely with an optional lock.

    Args:
        json_path: Path to the JSON file.
        lock: Thread lock for safe access.

    Returns:
        The parsed JSON data.
    """
    if lock is not None:
        lock.acquire(timeout=udef.LOCK_TIMEOUT)
    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_data: Union[Dict[str, Any], List[Any]] = json.load(json_file)
    finally:
        if lock is not None:
            lock.release()
    return json_data


def zip_directory(input_dir: str, output_zip: str) -> None:
    """
    Create a zip archive of a directory.

    Args:
        input_dir: Path to the directory to zip.
        output_zip: Path to save the zip archive to.
    """
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path: str = os.path.join(root, file)
                arcname: str = os.path.relpath(file_path, input_dir)
                zipf.write(file_path, arcname)
    print(f"Directory '{input_dir}' zipped to '{output_zip}'")


def extract_zip(input_zip: str, output_dir: str) -> None:
    """
    Extracts a zip archive to a directory.

    Args:
        input_zip: Path to the zip archive.
        output_dir: Directory where contents will be extracted.
    """
    with zipfile.ZipFile(input_zip, "r") as zipf:
        zipf.extractall(output_dir)
    print(f"Zip archive '{input_zip}' was extracted to directory '{output_dir}'")


def confirm_proceed(default_no: bool = True, operation_name: str = "update") -> bool:
    """
    Ask user for confirmation with timeout and default response
    Returns True if user confirms, False if user cancels or timeout
    """

    def timeout_handler(signum: int, frame: Any) -> None:
        """Raises TimeoutError on signal."""
        raise TimeoutError()

    try:
        # Check if signal.SIGALRM is available (Unix systems)
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(udef.CONFIRM_TIMEOUT)
            try:
                prompt: str = f"\nProceed with {operation_name}? (y/n) [{'n' if default_no else 'y'}] (timeout {udef.CONFIRM_TIMEOUT}s): "
                response: str = input(prompt).strip().lower()
                signal.alarm(0)
                if response in ["y", "yes"]:
                    return True
                elif response in ["n", "no"]:
                    return False
                else:
                    return not default_no
            except TimeoutError:
                print(f"\nTimeout reached. Defaulting to {'no' if default_no else 'yes'}.")
                return not default_no
            except KeyboardInterrupt:
                print(f"\nOperation interrupted by user. Defaulting to {'no' if default_no else 'yes'}.")
                return not default_no
        else:
            # Fallback for Windows
            prompt = f"\nProceed with {operation_name}? (y/n) [{'n' if default_no else 'y'}]: "
            response = input(prompt).strip().lower()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            return not default_no
    except Exception:
        return not default_no


def compose_dirpath(fun: T) -> T:
    """
    Decorator to compose directory paths based on home_dir, ndays, num_bins, etc.

    Args:
        fun: Function to decorate.

    Returns:
        Decorated function.
    """

    def inner(
        home_dir: str, ndays: int, nbins: Union[int, List[int]], output_dir: str, area: str, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Inner wrapper function.
        """
        if not isinstance(nbins, Iterable):
            dir_path: str = os.path.join(home_dir, "assets", output_dir, f"{ndays}_days", f"{area}_{nbins}")
            return fun(dir_path, *args, **kwargs)

        dir_paths: List[str] = []
        for gs in nbins:
            dir_paths.append(os.path.join(home_dir, "assets", output_dir, f"{ndays}_days", f"{area}_{gs}"))
        return fun(dir_paths, *args, **kwargs)

    return cast(T, inner)
