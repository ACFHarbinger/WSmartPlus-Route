"""__init__.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import __init__
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from logic.src.configs import Config
from logic.src.configs.tasks.eval import EvalConfig

from .engine import eval_dataset, eval_dataset_mp, get_best
from .validation import validate_eval_args


def run_evaluate_model(cfg: Config) -> None:
    """
    Main entry point for the evaluation script.

    Args:
        cfg: Root Hydra configuration with ``cfg.eval`` containing evaluation
            parameters.
    """
    from .validation import validate_eval_config

    # Validate and sanitize config values
    validate_eval_config(cfg)

    ev = cfg.eval

    random.seed(ev.seed)
    np.random.seed(ev.seed)
    torch.manual_seed(ev.seed)

    # Resolve strategy from decoding config
    strategy = ev.decoding.strategy if ev.decoding else "greedy"

    # Resolve beam widths
    beam_widths: List[int] = []
    bw = ev.decoding.beam_width if ev.decoding else None
    if bw is None:
        beam_widths = [0]
    elif isinstance(bw, (list, tuple)):
        beam_widths = list(bw)
    else:
        beam_widths = [bw]

    for beam_width in beam_widths:
        datasets = ev.datasets
        if not datasets:
            print("No datasets for evaluation found. Skipping evaluation.")
            return

        if not isinstance(datasets, (list, tuple)):
            datasets = [datasets]

        for dataset_path in datasets:
            # Extract temperature from decoding config
            temp = ev.decoding.temperature if ev.decoding else 1.0
            eval_dataset(dataset_path, beam_width, temp, cfg, strategy=strategy)


__all__ = ["run_evaluate_model", "eval_dataset", "eval_dataset_mp", "get_best", "validate_eval_args"]
