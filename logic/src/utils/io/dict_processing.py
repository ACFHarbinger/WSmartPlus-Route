"""
Dictionary and list processing utilities for JSON-like data.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from logic.src.interfaces import ITraversable


def process_dict_of_dicts(
    data_dict: Dict[str, Any],
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
    for k, v in data_dict.items():
        if isinstance(v, ITraversable) and process_dict_of_dicts(v, output_key, process_func, update_val):  # type: ignore[arg-type]
            modified = True
        elif k == output_key:
            if process_func:
                if isinstance(v, (list, tuple)):
                    data_dict[k] = type(v)([process_func(i, update_val) for i in v])
                else:
                    data_dict[k] = process_func(v, update_val)
            else:
                data_dict[k] = update_val
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
    """
    modified = False
    for item in data_list:
        if isinstance(item, dict) and process_dict_of_dicts(item, output_key, process_func, update_val):
            modified = True
    return modified


def process_dict_two_inputs(
    data_dict: Union[Dict[str, Any], Any],  # Accept ITraversable via duck typing
    input_key1: str,
    input_key2_or_val: Union[str, int, float, None],
    output_key: str,
    process_func: Callable[[Any, Any], Any],
) -> bool:
    """
    Process a dictionary using two inputs for a transformation.
    """
    modified = False
    # Check if this level has both keys
    has_k1 = input_key1 in data_dict
    val2 = None
    has_k2 = False

    if isinstance(input_key2_or_val, str):
        if input_key2_or_val in data_dict:
            has_k2 = True
            val2 = data_dict[input_key2_or_val]
    else:
        has_k2 = True
        val2 = input_key2_or_val

    if has_k1 and has_k2:
        val1 = data_dict[input_key1]
        if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)) and len(val1) == len(val2):
            data_dict[output_key] = type(val1)([process_func(v1, v2) for v1, v2 in zip(val1, val2)])
        else:
            data_dict[output_key] = process_func(val1, val2)
        modified = True

    # Recurse
    for v in data_dict.values():
        if isinstance(v, ITraversable) and process_dict_two_inputs(
            v,
            input_key1,
            input_key2_or_val,
            output_key,
            process_func,  # type: ignore[arg-type]
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
    """
    modified = False
    for item in data_list:
        if isinstance(item, ITraversable) and process_dict_two_inputs(
            item, input_key1, input_key2_or_val, output_key, process_func
        ):
            modified = True
    return modified
