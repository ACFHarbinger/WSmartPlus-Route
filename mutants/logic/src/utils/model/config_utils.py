"""
Configuration loading utilities for model arguments.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def load_args(filename: str) -> Dict[str, Any]:
    """
    Loads argument configuration from a JSON file.
    Handles deprecated keys for backward compatibility.

    Args:
        filename: Path to args.json.

    Returns:
        The loaded arguments.
    """
    with open(filename, "r") as f:
        args = json.load(f)

    # Backwards compatibility
    if "data_distribution" not in args:
        args["data_distribution"] = None
        probl, *dist = args["problem"].split("_")
        if probl in ("vrpp", "wcvrp"):
            args["problem"] = probl
            args["data_distribution"] = dist[0]

    if "aggregation_graph" not in args:
        args["aggregation_graph"] = "avg"
    return args
