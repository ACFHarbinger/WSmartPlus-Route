"""validators.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import validators
"""

import re
from typing import Any, Dict, Optional

from logic.src.constants import MAP_DEPOTS, WASTE_TYPES


def _sanitize_area(area: Optional[str]) -> str:
    if area is None:
        return "riomaior"
    area = re.sub(r"[^a-zA-Z]", "", area.lower())
    assert area in MAP_DEPOTS, f"Unknown area {area}, available areas: {list(MAP_DEPOTS.keys())}"
    return area


def _sanitize_waste(waste: Optional[str]) -> str:
    if waste is None:
        return "plastic"
    waste = re.sub(r"[^a-zA-Z]", "", waste.lower())
    assert waste in WASTE_TYPES, f"Unknown waste type {waste}, available types: {list(WASTE_TYPES.keys())}"
    return waste


def _get_graph_list(args: Dict[str, Any]) -> tuple[str, list[Dict[str, Any]]]:
    """Select correct graph list based on dataset_type."""
    dst_type = args.get("dataset_type", "train")
    if dst_type == "train":
        graphs = args.get("train_graphs", [])
    elif dst_type == "val":
        graphs = args.get("val_graphs", [])
    elif dst_type in ["test", "test_simulator"]:
        graphs = args.get("test_graphs", [])
    else:
        graphs = args.get("graphs", [])
    return dst_type, graphs


def _validate_filename_args(args: Dict[str, Any], dataset_count: int) -> None:
    """Validate filename constraints."""
    if "filename" not in args or args["filename"] is None:
        return

    is_single_problem = (isinstance(args.get("problem"), str) and args.get("problem") != "all") or (
        isinstance(args.get("problem"), list) and len(args["problem"]) == 1
    )

    assert is_single_problem and dataset_count <= 1, "Can only specify filename when generating a single dataset"


def _validate_problem_args(args: Dict[str, Any]) -> None:
    """Problem-specific validation."""
    if args.get("problem") in ["all", "swcvrp"]:
        assert args.get("mu") is not None, "Must specify mu when generating swcvrp datasets"
        assert args.get("sigma") is not None, "Must specify sigma when generating swcvrp datasets"
        assert len(args["mu"]) == len(args["sigma"]), "mu and sigma must have same length"


def validate_gen_data_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and post-process arguments for data generation."""
    args = args.copy()

    dst_type, graphs = _get_graph_list(args)
    dataset_count = len(graphs)

    _validate_filename_args(args, dataset_count)
    _validate_problem_args(args)

    # Process graphs
    for graph in graphs:
        graph["area"] = _sanitize_area(graph.get("area"))
        graph["waste_type"] = _sanitize_waste(graph.get("waste_type"))

        if graph.get("num_loc") is None:
            graph["num_loc"] = 50
        if graph.get("vertex_method") is None:
            graph["vertex_method"] = "mmn"
        if graph.get("focus_size") is None:
            graph["focus_size"] = 31

    # Update back into args
    key_map = {
        "train": "train_graphs",
        "val": "val_graphs",
        "test": "test_graphs",
        "test_simulator": "test_graphs",
    }
    target_key = key_map.get(dst_type, "graphs")
    args[target_key] = graphs

    return args
