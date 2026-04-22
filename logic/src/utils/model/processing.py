"""
Model processing and utility functions.

Attributes:
    get_inner_model: Returns the underlying model from a DataParallel wrapper if present.
    parse_softmax_temperature: Parses softmax temperature, supporting loading from a file (schedule) or a float.

Example:
    >>> from logic.src.utils.model.processing import get_inner_model
    >>> model = torch.nn.DataParallel(torch.nn.Linear(10, 2))
    >>> inner_model = get_inner_model(model)
    >>> isinstance(inner_model, torch.nn.Linear)
    True
    >>> from logic.src.utils.model.processing import parse_softmax_temperature
    >>> parse_softmax_temperature(1.0)
    1.0
    >>> parse_softmax_temperature("path/to/temp.txt")
    1.0
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np
import torch
from torch import nn


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
    if isinstance(raw_temp, str) and os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)
