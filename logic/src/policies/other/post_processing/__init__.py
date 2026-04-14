"""
Routing post-processing sub-package.
"""

from logic.src.interfaces import IPostProcessor

from .adaptive_large_neighborhood_search import AdaptiveLargeNeighborhoodSearchPostProcessor
from .base.factory import PostProcessorFactory
from .base.registry import PostProcessorRegistry
from .branch_and_price import BranchAndPricePostProcessor
from .cheapest_insertion import CheapestInsertionPostProcessor
from .cross_exchange import CrossExchangePostProcessor
from .dp_route_reopt import DPRouteReoptPostProcessor
from .fast_tsp import FastTSPPostProcessor
from .fix_and_optimize import FixAndOptimizePostProcessor
from .guided_local_search import GuidedLocalSearchPostProcessor
from .learned import LearnedPostProcessor
from .lkh import LinKernighanHelsgaunPostProcessor
from .local_search import ClassicalLocalSearchPostProcessor
from .node_exchange_steepest import NodeExchangeSteepestPostProcessor
from .or_opt import OrOptPostProcessor
from .or_opt_steepest import OrOptSteepestPostProcessor
from .path import PathPostProcessor
from .profitable_detour import ProfitableDetourPostProcessor
from .random_local_search import RandomLocalSearchPostProcessor
from .regret_k_insertion import RegretKInsertionPostProcessor
from .ruin_recreate import RuinRecreatePostProcessor
from .set_partitioning import SetPartitioningPostProcessor
from .set_partitioning_polish import SetPartitioningPolishPostProcessor
from .simulated_annealing import SimulatedAnnealingPostProcessor
from .steepest_two_opt import SteepestTwoOptPostProcessor
from .two_phase import TwoPhasePostProcessor

__all__ = [
    "IPostProcessor",
    "PostProcessorRegistry",
    "PostProcessorFactory",
    "FastTSPPostProcessor",
    "LinKernighanHelsgaunPostProcessor",
    "ClassicalLocalSearchPostProcessor",
    "RandomLocalSearchPostProcessor",
    "PathPostProcessor",
    "AdaptiveLargeNeighborhoodSearchPostProcessor",
    "CheapestInsertionPostProcessor",
    "CrossExchangePostProcessor",
    "GuidedLocalSearchPostProcessor",
    "OrOptPostProcessor",
    "ProfitableDetourPostProcessor",
    "RegretKInsertionPostProcessor",
    "RuinRecreatePostProcessor",
    "SimulatedAnnealingPostProcessor",
    "TwoPhasePostProcessor",
    "SteepestTwoOptPostProcessor",
    "OrOptSteepestPostProcessor",
    "NodeExchangeSteepestPostProcessor",
    "DPRouteReoptPostProcessor",
    "FixAndOptimizePostProcessor",
    "SetPartitioningPolishPostProcessor",
    "SetPartitioningPostProcessor",
    "BranchAndPricePostProcessor",
    "LearnedPostProcessor",
]
