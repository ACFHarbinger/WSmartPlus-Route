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
        Get IRI score for an operator.

        Args:
            operator: Operator name.
            operator_improvements: Recent improvements for this operator.

        Returns:
            Normalized IRI score (0.0-1.0).
        """
        if len(operator_improvements) == 0:
            return 0.5  # Neutral score for untried operators

        # Average improvement
        avg_improvement = np.mean(list(operator_improvements))

        # Normalize using global improvements
        if len(self.improvements) > 0:
            global_mean = np.mean(list(self.improvements))
            global_std = np.std(list(self.improvements))

            if global_std > 0:
                # Z-score normalization, then sigmoid to [0, 1]
                z_score = (avg_improvement - global_mean) / global_std
                return 1.0 / (1.0 + np.exp(-z_score))
            else:
                return 0.5 if avg_improvement == global_mean else (1.0 if avg_improvement > global_mean else 0.0)

        return 0.5


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
        Get TBI score for an operator.

        Args:
            operator: Operator name.
            operator_times: Recent execution times for this operator.

        Returns:
            Normalized TBI score (0.0-1.0), where higher is better (faster).
        """
        if len(operator_times) == 0:
            return 0.5  # Neutral score for untried operators

        # Average execution time
        avg_time = np.mean(list(operator_times))

        # Normalize using global times
        if len(self.times) > 0:
            global_mean = np.mean(list(self.times))
            global_std = np.std(list(self.times))

            if global_std > 0:
                # Z-score normalization (inverted: faster is better)
                z_score = (global_mean - avg_time) / global_std
                return 1.0 / (1.0 + np.exp(-z_score))
            else:
                return 0.5 if avg_time == global_mean else (1.0 if avg_time < global_mean else 0.0)

        return 0.5
