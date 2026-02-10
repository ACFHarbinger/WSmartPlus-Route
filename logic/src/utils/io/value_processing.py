"""
Recursive traversal utilities for JSON-like data structures.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from logic.src.interfaces import ITraversable


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

    if isinstance(data, ITraversable):
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


def _extract_two_vals(data: Any, k1: str, k2_or_val: Any) -> Tuple[Any, Any, bool]:
    """Extract v1 and v2 based on input keys/val."""
    if k1 not in data:
        return None, None, False
    v1 = data[k1]
    if isinstance(k2_or_val, str):
        if k2_or_val in data:
            return v1, data[k2_or_val], True
        return None, None, False
    return v1, k2_or_val, True


def _process_val_pair(path: str, v1: Any, v2: Any, results: List[Tuple[str, Any, Any]]):
    """Process a pair of values, handling list/tuple combinations."""
    if isinstance(v1, (list, tuple)):
        if isinstance(v2, (list, tuple)) and len(v1) == len(v2):
            for i, (item1, item2) in enumerate(zip(v1, v2)):
                results.append((f"{path}[{i}]" if path else f"[{i}]", item1, item2))
        else:
            for i, item1 in enumerate(v1):
                results.append((f"{path}[{i}]" if path else f"[{i}]", item1, v2))
    elif isinstance(v2, (list, tuple)):
        for i, item2 in enumerate(v2):
            results.append((f"{path}[{i}]" if path else f"[{i}]", v1, item2))
    else:
        results.append((path, v1, v2))


def find_two_input_values(
    data: Union[Dict[str, Any], List[Any]],
    current_path: str = "",
    input_key1: Optional[str] = None,
    input_key2: Union[str, int, float, None] = None,
) -> List[Tuple[str, Any, Any]]:
    """Recursively find all pairs of *source* values for two-input mode."""
    results = []

    if isinstance(data, ITraversable) and input_key1 is not None:
        v1, v2, success = _extract_two_vals(data, input_key1, input_key2)
        if success:
            _process_val_pair(current_path, v1, v2, results)

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
