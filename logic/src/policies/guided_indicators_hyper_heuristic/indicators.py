"""
Guidance Indicators for GIHH.

This module implements the two guidance indicators:
1. Improvement Rate Indicator (IRI): Measures solution quality improvement
2. Time-based Indicator (TBI): Measures computational efficiency
"""

from collections import deque
from typing import Deque

import numpy as np


class ImprovementRateIndicator:
    """
    Improvement Rate Indicator (IRI).

    Measures the average improvement achieved by each operator over a sliding window.
    Higher values indicate operators that consistently improve solution quality.
    """

    def __init__(self, window_size: int = 20):
        """
        Initialize IRI.

        Args:
            window_size: Number of recent improvements to track.
        """
        self.window_size = window_size
        self.improvements: Deque[float] = deque(maxlen=window_size)

    def update(self, operator: str, improvement: float) -> None:
        """
        Update IRI with new improvement value.

        Args:
            operator: Operator name (for future per-operator tracking).
            improvement: Improvement in objective value.
        """
        self.improvements.append(improvement)

    def get_score(self, operator: str, operator_improvements: Deque[float]) -> float:
        """
        Get IRI score for an operator (Chen 2018, Eq 4).
        Score is the average improvement rate of the operator.
        """
        if not operator_improvements:
            return 1.0  # Initial optimism

        return float(np.mean(operator_improvements))


class TimeBasedIndicator:
    """
    Time-based Indicator (TBI).

    Measures the computational efficiency of each operator.
    Lower execution times result in higher scores, encouraging fast operators.
    """

    def __init__(self, window_size: int = 20):
        """
        Initialize TBI.

        Args:
            window_size: Number of recent execution times to track.
        """
        self.window_size = window_size
        self.times: Deque[float] = deque(maxlen=window_size)

    def update(self, operator: str, elapsed_time: float) -> None:
        """
        Update TBI with new execution time.

        Args:
            operator: Operator name (for future per-operator tracking).
            elapsed_time: Time taken by operator in seconds.
        """
        self.times.append(elapsed_time)

    def get_score(self, operator: str, operator_times: Deque[float]) -> float:
        """
        Get TBI score for an operator (Chen 2018, Eq 5).
        TBI_i = 1 / average_time_i
        """
        if not operator_times:
            return 1.0  # Initial optimism

        avg_time = float(np.mean(operator_times))
        if avg_time < 1e-9:
            return 1e9
        return 1.0 / avg_time
