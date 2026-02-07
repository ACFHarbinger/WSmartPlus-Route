"""
Path manipulation utilities.
"""

from __future__ import annotations

import os
from typing import Optional


def get_path_until_string(path: str, end_str: str) -> Optional[str]:
    """
    Truncates a path up to the first occurrence of a specific directory component.

    Args:
        path: The full path.
        end_str: The directory name to truncate at.

    Returns:
        The truncated path or None if end_str is not found.
    """
    path_ls = str.split(path, os.sep)
    try:
        idx = path_ls.index(end_str)
        return os.sep.join(path_ls[: idx + 1])
    except ValueError:
        print(f"Path '{path}' does not contain '{end_str}'")
        return None
