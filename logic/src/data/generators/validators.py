"""validators.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import validators
"""

import re
from typing import Any, Dict

from logic.src.constants import MAP_DEPOTS, WASTE_TYPES


def validate_gen_data_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and post-process arguments for data generation.

    This function checks consistency between arguments, such as ensuring a
    filename is only provided when a single dataset is being generated.

    Args:
        args: Dictionary of parsed arguments.

    Returns:
        The validated and potentially modified dictionary of arguments.

    Raises:
        AssertionError: If incompatible arguments are provided (e.g., mu/sigma length mismatch).
    """
    args = args.copy()
    assert (
        "filename" not in args
        or args["filename"] is None
        or (
            (isinstance(args.get("problem"), str) and args.get("problem") != "all")
            or (isinstance(args.get("problem"), list) and len(args["problem"]) == 1)
        )
        and len(args.get("num_locs", [])) == 1
    ), "Can only specify filename when generating a single dataset"

    if args["problem"] in ["all", "swcvrp"]:
        assert "mu" in args and args["mu"] is not None, "Must specify mu when generating swcvrp datasets"
        assert "sigma" in args and args["sigma"] is not None, "Must specify sigma when generating swcvrp datasets"
        assert len(args["mu"]) == len(args["sigma"]), "Must specify same number of mu and sigma values"

    assert (
        "focus_graphs" not in args
        or args["focus_graphs"] is None
        or len(args["focus_graphs"]) == len(args.get("num_locs", []))
    )

    if "focus_graphs" not in args or args["focus_graphs"] is None:
        args["focus_graphs"] = [None] * len(args.get("num_locs", []))
    else:
        args["area"] = re.sub(r"[^a-zA-Z]", "", args.get("area", "").lower())
        assert args["area"] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(
            args["area"], MAP_DEPOTS.keys()
        )

    args["waste_type"] = re.sub(r"[^a-zA-Z]", "", args.get("waste_type", "").lower())
    assert args["waste_type"] in WASTE_TYPES.keys() or args["waste_type"] is None, (
        "Unknown waste type {}, available waste types: {}".format(args["waste_type"], WASTE_TYPES.keys())
    )
    return args
