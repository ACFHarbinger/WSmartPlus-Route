"""
Selection Strategy Implementations for WSmart-Route.

This package contains concrete implementations of the `MustGoSelectionStrategy`
interface. Strategies determine which bins must be collected on a given day
based on different criteria (fill levels, revenue, schedule, etc.).

Includes both:
- Single-instance selectors (for simulation): SelectionContext-based
- Vectorized selectors (for training): Batched PyTorch tensor operations

Attributes:
    CombinedSelector (class): Combines multiple selection strategies.
    LastMinuteSelector (class): Selects bins that are about to overflow.
    LookaheadSelector (class): Selects bins based on future fill predictions.
    ManagerSelector (class): Selects bins based on a manager agent's policy.
    RegularSelector (class): Selects bins based on fixed schedule frequency.
    RevenueSelector (class): Selects bins based on revenue potential.
    ServiceLevelSelector (class): Selects bins to maintain a service level.
    VectorizedSelector (class): Abstract base for vectorized selection.
    create_selector_from_config (function): Factory function for config-based creation.
    get_vectorized_selector (function): Factory function for vectorized selectors.

Example:
    >>> from logic.src.policies.must_go import create_selector_from_config
    >>> selector = create_selector_from_config(config)
    >>> selected_nodes = selector.select(context)
"""

from logic.src.interfaces import IMustGoSelectionStrategy
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

from .base import MustGoSelectionFactory, MustGoSelectionRegistry, SelectionContext

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
    "IMustGoSelectionStrategy",
    "MustGoSelectionFactory",
    "MustGoSelectionRegistry",
    "SelectionContext",
]
