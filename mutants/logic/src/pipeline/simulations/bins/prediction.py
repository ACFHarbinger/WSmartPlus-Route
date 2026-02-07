"""
Statistical prediction and frequency calculation for waste bins.
"""

from typing import Tuple

import numpy as np
from scipy.stats import gamma


def predict_days_to_overflow(ui: np.ndarray, vi: np.ndarray, f: np.ndarray, cl: float) -> np.ndarray:
    """
    Internal math for predicting days until a bin overflows.

    Uses the Gamma distribution CDF to estimate the probability of
    reaching 100% capacity within a 31-day window.

    Args:
        ui: Mean fill rate.
        vi: Variance of fill rate.
        f: Current fill level.
        cl: Confidence level (0-1).

    Returns:
        np.ndarray: Predicted days to overflow per bin (clipped at 31).
    """
    n = np.zeros(ui.shape[0]) + 31
    for ii in np.arange(1, 31, 1):
        # Prevent division by zero
        safe_vi = np.where(vi == 0, 1e-9, vi)
        safe_ui = np.where(ui == 0, 1e-9, ui)

        k = ii * safe_ui**2 / safe_vi
        th = safe_vi / safe_ui
        aux = np.zeros(ui.shape[0]) + 31

        # Calculate CDF
        p = 1 - gamma.cdf(100 - f, k, scale=th)

        aux[np.nonzero(p > cl)[0]] = ii
        n = np.minimum(n, aux)
        if np.all(p > cl):
            return n
    return n


def calculate_frequency_and_level(ui: float, vi: float, cf: float) -> Tuple[int, float]:
    """
    Calculates the recommended visit frequency and target overflow level.

    Args:
        ui: Mean daily fill rate.
        vi: Variance of daily fill rate.
        cf: Target confidence level (e.g., 0.9 for 90% service level).

    Returns:
        Tuple[int, float]: (Optimal days between visits, Target level at visit).
    """
    ov = 80.0  # Default fallback
    for n in range(1, 50):
        # Prevent division by zero
        safe_vi = 1e-9 if vi == 0 else vi
        safe_ui = 1e-9 if ui == 0 else ui

        k = n * safe_ui**2 / safe_vi
        th = safe_vi / safe_ui

        if n == 1:
            ov = 100 - gamma.ppf(1 - cf, k, scale=th)

        v = gamma.ppf(1 - cf, k, scale=th)
        if v > 100:
            return n, ov
    return 49, ov
