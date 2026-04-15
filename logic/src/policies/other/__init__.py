"""
Other specialized policies for WSmart-Route.

Includes policies that do not fit into the standard RL or classical categories,
such as baseline comparisons and experimental heuristics.
"""

from .mandatory import (
    CombinedSelector,
    DeadlineDrivenSelection,
    IMandatorySelectionStrategy,
    LastMinuteSelector,
    LookaheadSelector,
    ManagerSelector,
    MandatorySelectionFactory,
    MandatorySelectionRegistry,
    MultiDayOverflowSelection,
    ParetoFrontSelection,
    ProfitPerKmSelection,
    RegularSelector,
    RevenueSelector,
    SelectionContext,
    ServiceLevelSelector,
    SpatialSynergySelection,
    StochasticRegretSelection,
    VectorizedSelector,
    create_selector_from_config,
    get_vectorized_selector,
)
from .route_improvement import (
    ClassicalLocalSearchRouteImprover,
    FastTSPRouteImprover,
    IRouteImprovement,
    PathRouteImprover,
    RandomLocalSearchRouteImprover,
    RouteImproverFactory,
    RouteImproverRegistry,
)

__all__ = [
    "IMandatorySelectionStrategy",
    "CombinedSelector",
    "LastMinuteSelector",
    "LookaheadSelector",
    "ManagerSelector",
    "RegularSelector",
    "RevenueSelector",
    "ServiceLevelSelector",
    "VectorizedSelector",
    "create_selector_from_config",
    "get_vectorized_selector",
    # New simulation strategies
    "DeadlineDrivenSelection",
    "MultiDayOverflowSelection",
    "ParetoFrontSelection",
    "ProfitPerKmSelection",
    "SpatialSynergySelection",
    "StochasticRegretSelection",
    "ClassicalLocalSearchRouteImprover",
    "FastTSPRouteImprover",
    "IRouteImprovement",
    "PathRouteImprover",
    "RouteImproverFactory",
    "RouteImproverRegistry",
    "RandomLocalSearchRouteImprover",
    "MandatorySelectionFactory",
    "MandatorySelectionRegistry",
    "SelectionContext",
]
