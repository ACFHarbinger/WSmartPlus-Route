"""
Guidance Indicators for GIHH.

This module implements the two guidance indicators used by the hyper-heuristic
to select Low-Level Heuristics (LLHs) based on their historical performance:

1. Improvement Rate Indicator (IRI): Measures solution quality improvement
2. Time-based Indicator (TBI): Measures computational efficiency

Reference:
    Chen, B., Qu, R., Bai, R., & Laesanklang, W. (2018). "A hyper-heuristic with
    two guidance indicators for bi-objective mixed-shift vehicle routing problem
    with time windows." European Journal of Operational Research, 269(2), 661-675.
"""

from collections import deque
from typing import Deque, Dict

import numpy as np


class ImprovementRateIndicator:
    """
    Improvement Rate Indicator (IRI).

    Measures the average improvement achieved by each operator over a sliding window.
    Higher values indicate operators that consistently improve solution quality.

    Each operator maintains its own history, allowing the hyper-heuristic to compare
    the relative effectiveness of different operators.

    Reference:
        Chen et al. (2018), Equation 4: IRI_i = (1/|W|) * Σ improvement_i
    """

    def __init__(self, window_size: int = 20):
        """
        Initialize IRI with per-operator tracking.

        Args:
            window_size: Number of recent improvements to track per operator.
        """
        self.window_size = window_size
        self.improvements: Dict[str, Deque[float]] = {}

    def update(self, operator: str, improvement: float) -> None:
        """
        Update IRI with new improvement value for a specific operator.

        Args:
            operator: Name of the operator that was applied.
            improvement: Change in objective value (positive = improvement).
        """
        if operator not in self.improvements:
            self.improvements[operator] = deque(maxlen=self.window_size)
        self.improvements[operator].append(improvement)

    def get_score(self, operator: str) -> float:
        """
        Get IRI score for an operator (Chen et al. 2018, Eq 4).

        Score is the average improvement rate of the operator over its history.
        Higher scores indicate operators that consistently improve solutions.

        Args:
            operator: Name of the operator to score.

        Returns:
            Average improvement rate. Returns 1.0 for operators with no history
            (initial optimism bias).
        """
        if operator not in self.improvements or not self.improvements[operator]:
            return 1.0  # Initial optimism for unexplored operators

        return float(np.mean(self.improvements[operator]))


class TimeBasedIndicator:
    """
    Time-based Indicator (TBI).

    Measures the computational efficiency of each operator. Lower execution times
    result in higher scores, encouraging the selection of fast operators.

    Each operator maintains its own execution time history for independent scoring.

    Reference:
        Chen et al. (2018), Equation 5: TBI_i = 1 / avg_time_i
    """

    def __init__(self, window_size: int = 20):
        """
        Initialize TBI with per-operator tracking.

        Args:
            window_size: Number of recent execution times to track per operator.
        """
        self.window_size = window_size
        self.times: Dict[str, Deque[float]] = {}

    def update(self, operator: str, elapsed_time: float) -> None:
        """
        Update TBI with new execution time for a specific operator.

        Args:
            operator: Name of the operator that was applied.
            elapsed_time: Time taken by operator in seconds.
        """
        if operator not in self.times:
            self.times[operator] = deque(maxlen=self.window_size)
        self.times[operator].append(elapsed_time)

    def get_score(self, operator: str) -> float:
        """
        Get TBI score for an operator (Chen et al. 2018, Eq 5).

        Score is the inverse of average execution time. Faster operators receive
        higher scores, encouraging computational efficiency.

        Args:
            operator: Name of the operator to score.

        Returns:
            Inverse of average execution time. Returns 1.0 for operators with no
            history (initial optimism bias).
        """
        if operator not in self.times or not self.times[operator]:
            return 1.0  # Initial optimism for unexplored operators

        avg_time = float(np.mean(self.times[operator]))
        if avg_time < 1e-9:
            return 1e9  # Extremely fast operator
        return 1.0 / avg_time
