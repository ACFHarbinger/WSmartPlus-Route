"""__init__.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import __init__
"""

import random
from typing import Any, Dict

import numpy as np
import torch

from .engine import eval_dataset, eval_dataset_mp, get_best
from .validation import validate_eval_args


def run_evaluate_model(opts: Dict[str, Any]) -> None:
    """
    Main entry point for the evaluation script.
    """
    random.seed(opts["seed"])
    np.random.seed(opts["seed"])
    torch.manual_seed(opts["seed"])

    # Remap keys for compatibility with evaluation engine
    if "num_loc" in opts and "graph_size" not in opts:
        opts["graph_size"] = opts["num_loc"]
    if "load_path" in opts and "model" not in opts:
        opts["model"] = opts["load_path"]
    # Ensure beam_widths is iterable (can be int or list[int])
    beam_widths = opts.get("beam_width")
    if beam_widths is None:
        # Check nested decoding config if not at top level
        beam_widths = opts.get("decoding", {}).get("beam_width", [0])

    if not isinstance(beam_widths, (list, tuple)):
        beam_widths = [beam_widths]

    for beam_width in beam_widths:
        datasets = opts.get("datasets")
        if datasets is None:
            # Check nested eval config if not at top level
            datasets = opts.get("eval", {}).get("datasets")

        if not datasets:
            print("No datasets for evaluation found. Skipping evaluation.")
            return

        if not isinstance(datasets, (list, tuple)):
            datasets = [datasets]

        for dataset_path in datasets:
            # Extract temperature from decoding config or default to 1.0
            temp = opts.get("decoding", {}).get("temperature")
            if temp is None:
                temp = opts.get("eval", {}).get("decoding", {}).get("temperature", 1.0)
            eval_dataset(dataset_path, beam_width, temp, opts)


__all__ = ["run_evaluate_model", "eval_dataset", "eval_dataset_mp", "get_best", "validate_eval_args"]
