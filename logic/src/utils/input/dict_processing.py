"""
Dictionary and list processing utilities for JSON-like data.

Attributes:
    process_dict_of_dicts: Process a dictionary of dictionaries and apply a function to key values.
    process_list_of_dicts: Process a list of dictionaries.
    process_dict_two_inputs: Process a dictionary using two input keys for a transformation.
    process_list_two_inputs: Process a list of dictionaries using two input keys.

Example:
    >>> from logic.src.utils.io.dict_processing import process_dict_of_dicts, process_list_of_dicts, process_dict_two_inputs, process_list_two_inputs
    >>> processed = process_dict_of_dicts(data, "km", lambda x, y: x + y, 10)
    >>> processed = process_list_of_dicts(data, "km", lambda x, y: x + y, 10)
    >>> processed = process_dict_two_inputs(data, "key1", "key2", "output_key", lambda x, y: x + y)
    >>> processed = process_list_two_inputs(data, "key1", "key2", "output_key", lambda x, y: x + y)
"""

from typing import Any, Callable, Dict, List, Optional, Union, cast

from logic.src.interfaces.traversable import ITraversable


def process_dict_of_dicts(
    data_dict: object,
    output_key: str = "km",
    process_func: Optional[Callable[[Any, Any], Any]] = None,
    update_val: Union[int, float] = 0,
) -> bool:
    """
    Process a dictionary of dictionaries and apply process_func to key values.

    Args:
        data_dict: The dictionary to process.
        output_key: Key to target.
        process_func: Transformation function.
        update_val: Constant parameter for process_func.

    Returns:
        True if modified.
    """
    modified = False
    if isinstance(data_dict, ITraversable):
        # Cast to Any to allow __setitem__ if it supports it at runtime (e.g. dict)
        data_any = cast(Any, data_dict)
        for k, v in data_any.items():
            item_v: object = v
            if isinstance(item_v, (dict, list, ITraversable)) and process_dict_of_dicts(
                item_v, output_key, process_func, update_val
            ):
                modified = True
            elif k == output_key:
                if process_func:
                    if isinstance(v, (list, tuple)):
                        data_any[k] = type(v)([process_func(i, update_val) for i in v])
                    else:
                        data_any[k] = process_func(v, update_val)
                else:
                    data_any[k] = update_val
                modified = True
    return modified


def process_list_of_dicts(
    data_list: List[Dict[str, Any]],
    output_key: str = "km",
    process_func: Optional[Callable[[Any, Any], Any]] = None,
    update_val: Union[int, float] = 0,
) -> bool:
    """
    Process a list of dictionaries.

    Args:
        data_list: The list of dictionaries to process.
        output_key: Key to target.
        process_func: Transformation function.
        update_val: Constant parameter for process_func.

    Returns:
        True if modified.
    """
    modified = False
    for item in data_list:
        item_obj: object = item
        if isinstance(item_obj, (dict, ITraversable)) and process_dict_of_dicts(
            item_obj, output_key, process_func, update_val
        ):
            modified = True
    return modified


def process_dict_two_inputs(
    data_dict: object,
    input_key1: str,
    input_key2_or_val: Union[str, int, float, None],
    output_key: str,
    process_func: Callable[[Any, Any], Any],
) -> bool:
    """
    Process a dictionary using two inputs for a transformation.

    Args:
        data_dict: The dictionary to process.
        input_key1: The first input key.
        input_key2_or_val: The second input key or a constant value.
        output_key: The output key.
        process_func: The transformation function.

    Returns:
        True if the dictionary was modified, False otherwise.
    """
    modified = False
    if not isinstance(data_dict, ITraversable):
        return False

    # Cast to Any to allow get and __setitem__
    data_any = cast(Any, data_dict)

    # Check if this level has both keys
    val1 = data_any.get(input_key1, None)
    has_k1 = val1 is not None

    val2 = None
    has_k2 = False

    if isinstance(input_key2_or_val, str):
        val2 = data_any.get(input_key2_or_val, None)
        if val2 is not None:
            has_k2 = True
    else:
        has_k2 = True
        val2 = input_key2_or_val

    if has_k1 and has_k2:
        if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)) and len(val1) == len(val2):
            data_any[output_key] = type(val1)([process_func(v1, v2) for v1, v2 in zip(val1, val2, strict=False)])
        else:
            data_any[output_key] = process_func(val1, val2)
        modified = True

    # Recurse
    # Use items() for safer traversal if available, otherwise fallback
    if hasattr(data_dict, "values"):
        # Cast to Any to iterate over values safely
        vals = cast(Any, data_dict).values()
        for v in vals:
            item_v: object = v
            if isinstance(item_v, (dict, list, ITraversable)) and process_dict_two_inputs(
                item_v,
                input_key1,
                input_key2_or_val,
                output_key,
                process_func,
            ):
                modified = True
    return modified


def process_list_two_inputs(
    data_list: List[Dict[str, Any]],
    input_key1: str,
    input_key2_or_val: Union[str, int, float],
    output_key: str,
    process_func: Callable[[Any, Any], Any],
) -> bool:
    """
    Process a list of dictionaries using two inputs.

    Args:
        data_list: The list of dictionaries to process.
        input_key1: The first input key.
        input_key2_or_val: The second input key or a constant value.
        output_key: The output key.
        process_func: The transformation function.

    Returns:
        True if the list was modified, False otherwise.
    """
    modified = False
    for item in data_list:
        list_item: object = item
        if isinstance(list_item, ITraversable) and process_dict_two_inputs(
            list_item, input_key1, input_key2_or_val, output_key, process_func
        ):
            modified = True
    return modified
