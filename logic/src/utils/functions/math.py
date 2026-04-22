"""
Mathematical utilities for robust computations.

Attributes:
    safe_exp: Compute exponent with overflow/underflow protection.

Example:
    >>> from logic.src.utils.functions import safe_exp
    >>> safe_exp(1000)
    inf
    >>> safe_exp(-1000)
    0.0
"""

import math


def safe_exp(x: float) -> float:
    """
    Compute exponent with overflow/underflow protection.

    Args:
        x: Input value.

    Returns:
        float: math.exp(x) if safe, else 0.0 or inf.
    """
    try:
        # Typical float64 upper limit for exp is ~709
        if x > 700:
            return float("inf")
        if x < -700:
            return 0.0
        return math.exp(x)
    except OverflowError:
        return float("inf") if x > 0 else 0.0
