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
    widths = opts["width"] if opts["width"] is not None else [0]
    for width in widths:
        for dataset_path in opts["datasets"]:
            eval_dataset(dataset_path, width, opts["softmax_temperature"], opts)


__all__ = ["run_evaluate_model", "eval_dataset", "eval_dataset_mp", "get_best", "validate_eval_args"]
