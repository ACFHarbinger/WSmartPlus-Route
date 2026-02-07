"""
Model processing and utility functions.
"""

from __future__ import annotations

import os
from typing import Union

import torch
import torch.nn as nn


def get_inner_model(model: nn.Module) -> nn.Module:
    """
    Returns the underlying model from a DataParallel wrapper if present.

    Args:
        model: The model (potentially wrapped).

    Returns:
        The inner model.
    """
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def parse_softmax_temperature(raw_temp: Union[str, float]) -> float:
    """
    Parses softmax temperature, supporting loading from a file (schedule) or a float.

    Args:
        raw_temp: The raw temperature argument.

    Returns:
        The parsed temperature.
    """
    import numpy as np

    if isinstance(raw_temp, str) and os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)
