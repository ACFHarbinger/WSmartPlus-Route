"""
Recursive traversal utilities for JSON-like data structures.

Attributes:
    find_single_input_values: Recursively find all *source* values for single-input mode.
    find_two_input_values: Recursively find all pairs of *source* values for two-input mode.

Example:
    >>> from logic.src.utils.io.value_processing import find_single_input_values, find_two_input_values
    >>> data = {"km": [10, 20], "kg": [100, 200]}
    >>> find_single_input_values(data)
    [('km', [10, 20])]
    >>> find_two_input_values(data, "km", "kg")
    [('km', [10, 20], [100, 200])]
"""

from typing import Any, List, Optional, Tuple, Union

from logic.src.interfaces.traversable import ITraversable


def find_single_input_values(
    data: object,
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
            item_v: object = v
            new_path = f"{current_path}.{k}" if current_path else k
            if k == output_key:
                results.append((current_path if current_path else k, item_v))
            elif isinstance(item_v, (dict, list, ITraversable)):
                results.extend(find_single_input_values(item_v, new_path, output_key))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            list_item: object = v
            new_path = f"{current_path}[{i}]"
            if isinstance(list_item, (dict, list, ITraversable)):
                results.extend(find_single_input_values(list_item, new_path, output_key))

    return results


def _extract_two_vals(data: Any, k1: str, k2_or_val: Any) -> Tuple[Any, Any, bool]:
    """
    Extract v1 and v2 based on input keys/val.

    Args:
        data: The data structure to search.
        k1: The first key to look for.
        k2_or_val: The second key or value to look for.

    Returns:
        Tuple of (v1, v2, success).
    """
    if k1 not in data:
        return None, None, False
    v1 = data[k1]
    if isinstance(k2_or_val, str):
        if k2_or_val in data:
            return v1, data[k2_or_val], True
        return None, None, False
    return v1, k2_or_val, True


def _process_val_pair(path: str, v1: Any, v2: Any, results: List[Tuple[str, Any, Any]]):
    """
    Process a pair of values, handling list/tuple combinations.

    Args:
        path: The path to the values.
        v1: The first value.
        v2: The second value.
        results: The list to append the results to.
    """
    if isinstance(v1, (list, tuple)):
        if isinstance(v2, (list, tuple)) and len(v1) == len(v2):
            for i, (item1, item2) in enumerate(zip(v1, v2, strict=False)):
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
    data: object,
    current_path: str = "",
    input_key1: Optional[str] = None,
    input_key2: Union[str, int, float, None] = None,
) -> List[Tuple[str, Any, Any]]:
    """
    Recursively find all pairs of *source* values for two-input mode.

    Args:
        data: The data structure to search.
        current_path: Current path in the recursive search.
        input_key1: The first key to look for.
        input_key2: The second key or value to look for.

    Returns:
        List of (location_path, value1, value2) tuples.
    """
    results: List[Tuple[str, Any, Any]] = []

    if isinstance(data, ITraversable) and input_key1 is not None:
        v1, v2, success = _extract_two_vals(data, input_key1, input_key2)
        if success:
            _process_val_pair(current_path, v1, v2, results)

        for k, v in data.items():
            item_v: object = v
            new_path = f"{current_path}.{k}" if current_path else k
            if isinstance(item_v, (dict, list, ITraversable)):
                results.extend(find_two_input_values(item_v, new_path, input_key1, input_key2))

    elif isinstance(data, list):
        for i, v in enumerate(data):
            list_item: object = v
            new_path = f"{current_path}[{i}]"
            if isinstance(list_item, (dict, list, ITraversable)):
                results.extend(find_two_input_values(list_item, new_path, input_key1, input_key2))

    return results
