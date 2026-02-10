"""
Validation logic for simulation testing pipeline.
"""

import re
from multiprocessing import cpu_count
from typing import Any, Dict

from logic.src.constants import MAP_DEPOTS, WASTE_TYPES


def validate_test_sim_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and post-processes arguments for test_sim.
    """
    args = args.copy()
    assert args.get("days", 0) >= 1, "Must run the simulation for 1 or more days"
    assert args.get("n_samples", 0) > 0, "Number of samples must be non-negative integer"

    args["area"] = re.sub(r"[^a-zA-Z]", "", args.get("area", "").lower())
    assert args["area"] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(
        args["area"], MAP_DEPOTS.keys()
    )

    args["waste_type"] = re.sub(r"[^a-zA-Z]", "", args.get("waste_type", "").lower())
    assert args["waste_type"] in WASTE_TYPES.keys() or args["waste_type"] is None, (
        "Unknown waste type {}, available waste types: {}".format(args["waste_type"], WASTE_TYPES.keys())
    )

    args["edge_threshold"] = (
        float(args["edge_threshold"])
        if "." in str(args.get("edge_threshold", "0"))
        else int(args.get("edge_threshold", "0"))
    )

    assert args.get("cpu_cores", 0) >= 0, "Number of CPU cores must be non-negative integer"
    assert args.get("cpu_cores", 0) <= cpu_count(), "Number of CPU cores to use cannot exceed system specifications"
    if args.get("cpu_cores") == 0:
        args["cpu_cores"] = cpu_count()

    return args
