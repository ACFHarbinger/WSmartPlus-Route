"""
Must-Go Selection Config module.

Configures the vectorized selection strategy used during training
to determine which bins must be collected.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MustGoConfig:
    """Configuration for must-go bin selection during training.

    The must-go selector determines which bins are mandatory collection
    targets during training, enforcing routing constraints based on
    fill levels, predictions, or other criteria.

    Attributes:
        strategy: Name of the selection strategy. Options:
            - "last_minute": Threshold-based (collect when fill > threshold)
            - "regular": Periodic collection on scheduled days
            - "lookahead": Predictive (collect if overflow within N days)
            - "revenue": Revenue-based (collect if revenue exceeds threshold)
            - "service_level": Statistical overflow prediction
            - "combined": Combine multiple strategies with OR logic
            - None: No must-go constraint (default behavior)
        threshold: Fill level threshold for last_minute strategy (0-1).
        frequency: Collection frequency for regular strategy (days).
        lookahead_days: Number of days to look ahead for lookahead strategy.
        confidence_factor: Standard deviations for service_level strategy.
        revenue_kg: Revenue per kg for revenue strategy.
        bin_capacity: Bin capacity for revenue calculation.
        revenue_threshold: Minimum revenue threshold for revenue strategy.
        max_fill: Maximum fill level (overflow threshold).
        combined_strategies: List of strategy configs for combined selector.
        params: Additional strategy-specific parameters.
    """

    strategy: Optional[str] = None

    # LastMinute parameters
    threshold: float = 0.7

    # Regular parameters
    frequency: int = 3

    # Lookahead parameters
    lookahead_days: int = 1

    # ServiceLevel parameters
    confidence_factor: float = 1.0

    # Revenue parameters
    revenue_kg: float = 1.0
    bin_capacity: float = 1.0
    revenue_threshold: float = 0.0

    # Common parameters
    max_fill: float = 1.0

    # Combined strategy configs (list of dicts with strategy configs)
    combined_strategies: Optional[list] = None

    # Additional parameters
    params: Dict[str, Any] = field(default_factory=dict)
