"""
Validation logic for evaluation pipeline.
"""

import re
from typing import Any, Dict

from logic.src.constants import MAP_DEPOTS, WASTE_TYPES


def validate_eval_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and post-processes arguments for eval.
    """
    args = args.copy()
    # Handle the -o alias for output_filename
    if "output_filename" in args and args["output_filename"] is not None:
        args["o"] = args["output_filename"]

    assert (
        "o" not in args
        or args["o"] is None
        or (len(args.get("datasets") or []) == 1 and len(args.get("width") or []) <= 1)
    ), "Cannot specify result filename with more than one dataset or more than one width"

    args["area"] = re.sub(r"[^a-zA-Z]", "", args.get("area", "").lower())
    assert args["area"] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(
        args["area"], MAP_DEPOTS.keys()
    )

    args["waste_type"] = re.sub(r"[^a-zA-Z]", "", args.get("waste_type", "").lower())
    assert (
        args["waste_type"] in WASTE_TYPES.keys() or args["waste_type"] is None
    ), "Unknown waste type {}, available waste types: {}".format(args["waste_type"], WASTE_TYPES.keys())
    return args
