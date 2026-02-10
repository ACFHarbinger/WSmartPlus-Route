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
    beam_widths = opts["beam_width"] if opts.get("beam_width") is not None else [0]
    for beam_width in beam_widths:
        for dataset_path in opts["datasets"]:
            # Extract temperature from decoding config or default to 1.0
            temp = opts.get("decoding", {}).get("temperature", 1.0)
            eval_dataset(dataset_path, beam_width, temp, opts)


__all__ = ["run_evaluate_model", "eval_dataset", "eval_dataset_mp", "get_best", "validate_eval_args"]
