"""
Context Package for Functional State Tracking.

Provides the typed ``SearchContext`` ledger and associated ``TypedDict``
metrics that flow immutably through the three-phase pipeline:

  Phase 1 — Mandatory Selection  : creates ``SearchContext``
  Phase 2 — Route Construction   : merges ``ConstructionMetrics`` / ``AcceptanceMetrics``
  Phase 3 — Route Improvement    : appends ``ImprovementMetrics``

Attributes:
    AcceptanceMetrics: Metrics tracked during route acceptance.
    ConstructionMetrics: Metrics tracked during route construction.
    ImprovementMetrics: Metrics tracked during route improvement.
    JointSelectionConstructionContext: Context for joint selection and construction.
    MultiDayContext: Context for multi-day planning.
    SearchContext: Context for search.
    SelectionContext: Context for selection.
    SearchPhase: Phase of the search.
    SelectionMetrics: Metrics for selection.
    merge_context: Function to merge contexts.

Example:
  >>> from logic.src.policies.context import SearchContext
  >>> search_context = SearchContext()
  >>> search_context = merge_context(search_context, construction_metrics)
  >>> search_context = merge_context(search_context, acceptance_metrics)
  >>> search_context = merge_context(search_context, improvement_metrics)
"""

from logic.src.interfaces.context import (
    AcceptanceMetrics,
    ConstructionMetrics,
    ImprovementMetrics,
    JointSelectionConstructionContext,
    MultiDayContext,
    SearchContext,
    SelectionContext,
    SearchPhase,
    SelectionMetrics,
    merge_context,
)

__all__ = [
    "SearchContext",
    "SearchPhase",
    "SelectionMetrics",
    "SelectionContext",
    "ConstructionMetrics",
    "AcceptanceMetrics",
    "ImprovementMetrics",
    "merge_context",
    "MultiDayContext",
    "JointSelectionConstructionContext",
]
