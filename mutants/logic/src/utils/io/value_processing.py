"""
Recursive traversal utilities for JSON-like data structures.
"""

from typing import Any, Dict, List, Optional, Tuple, Union


def find_single_input_values(
    data: Union[Dict[str, Any], List[Any]],
    current_path: str = "",
    output_key: str = "km",
) -> List[Tuple[str, Any]]:
    """
    Recursively find all *source* values for single-input mode.

    Args:
        data: The data structure to search.
        current_path: Current path in the recursive search.
        output_key: The key to look for.

    Returns:
        List of (location_path, value) tuples.
    """
    results = []

    if isinstance(data, dict):
        for k, v in data.items():
            new_path = f"{current_path}.{k}" if current_path else k
            if k == output_key:
                results.append((current_path if current_path else k, v))
            elif isinstance(v, (dict, list)):
                results.extend(find_single_input_values(v, new_path, output_key))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_path = f"{current_path}[{i}]"
            if isinstance(v, (dict, list)):
                results.extend(find_single_input_values(v, new_path, output_key))

    return results


def find_two_input_values(
    data: Union[Dict[str, Any], List[Any]],
    current_path: str = "",
    input_key1: Optional[str] = None,
    input_key2: Union[str, int, float, None] = None,
) -> List[Tuple[str, Any, Any]]:
    """
    Recursively find all pairs of *source* values for two-input mode.

    Args:
        data: The data structure to search.
        current_path: Current path string.
        input_key1: First key to find.
        input_key2: Second key to find (or constant value).

    Returns:
        List of (location_path, value1, value2) tuples.
    """
    results = []

    if isinstance(data, dict):
        has_k1 = input_key1 in data
        val2_direct = None
        has_k2 = False

        if isinstance(input_key2, str):
            has_k2 = input_key2 in data
            if has_k2:
                val2_direct = data[input_key2]
        else:
            has_k2 = True
            val2_direct = input_key2

        if has_k1 and has_k2:
            val1 = data[input_key1]
            if isinstance(val1, (list, tuple)):
                if isinstance(val2_direct, (list, tuple)) and len(val1) == len(val2_direct):
                    for i, (v1, v2) in enumerate(zip(val1, val2_direct)):
                        results.append((f"{current_path}[{i}]" if current_path else f"[{i}]", v1, v2))
                else:
                    for i, v1 in enumerate(val1):
                        results.append((f"{current_path}[{i}]" if current_path else f"[{i}]", v1, val2_direct))
            elif isinstance(val2_direct, (list, tuple)):
                for i, v2 in enumerate(val2_direct):
                    results.append((f"{current_path}[{i}]" if current_path else f"[{i}]", val1, v2))
            else:
                results.append((current_path, val1, val2_direct))

        for k, v in data.items():
            new_path = f"{current_path}.{k}" if current_path else k
            if isinstance(v, (dict, list)):
                results.extend(find_two_input_values(v, new_path, input_key1, input_key2))

    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_path = f"{current_path}[{i}]"
            if isinstance(v, (dict, list)):
                results.extend(find_two_input_values(v, new_path, input_key1, input_key2))

    return results
