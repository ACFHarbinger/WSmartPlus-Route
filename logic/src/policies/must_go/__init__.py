"""
Selection Strategy Implementations for WSmart-Route.

This package contains concrete implementations of the `MustGoSelectionStrategy`
interface. Strategies determine which bins must be collected on a given day
based on different criteria (fill levels, revenue, schedule, etc.).

Includes both:
- Single-instance selectors (for simulation): SelectionContext-based
- Vectorized selectors (for training): Batched PyTorch tensor operations
"""

from logic.src.models.policies.selection import (
    CombinedSelector,
    LastMinuteSelector,
    LookaheadSelector,
    ManagerSelector,
    RegularSelector,
    RevenueSelector,
    ServiceLevelSelector,
    VectorizedSelector,
    create_selector_from_config,
    get_vectorized_selector,
)

__all__ = [
    # Vectorized selectors for training
    "VectorizedSelector",
    "LastMinuteSelector",
    "RegularSelector",
    "LookaheadSelector",
    "RevenueSelector",
    "ServiceLevelSelector",
    "CombinedSelector",
    "ManagerSelector",
    "get_vectorized_selector",
    "create_selector_from_config",
]
