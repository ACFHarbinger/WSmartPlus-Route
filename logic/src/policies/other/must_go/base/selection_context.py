"""
Selection Context Module.

This module defines the `SelectionContext` data class, which serves as a
container for all relevant data needed by selection strategies to make decisions.

Attributes:
    None

Example:
    >>> from logic.src.policies.must_go.base.selection_context import SelectionContext
    >>> ctx = SelectionContext(bin_ids=np.array([1, 2]), current_fill=np.array([0.8, 0.5]))
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class SelectionContext:
    """
    Context container for all potential inputs required by selection strategies.
    """

    bin_ids: NDArray[np.int32]
    current_fill: NDArray[np.float64]
    accumulation_rates: Optional[NDArray[np.float64]] = None
    std_deviations: Optional[NDArray[np.float64]] = None
    current_day: int = 0
    threshold: float = 0.0
    next_collection_day: Optional[int] = None
    distance_matrix: Optional[NDArray[Any]] = None
    paths_between_states: Optional[List[List[List[int]]]] = None
    vehicle_capacity: float = 0.0
    revenue_kg: float = 0.0
    bin_density: float = 0.0
    bin_volume: float = 0.0
    max_fill: float = 100.0
