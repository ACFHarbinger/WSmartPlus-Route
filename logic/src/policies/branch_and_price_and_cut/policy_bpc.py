"""
BPC Policy Adapter.

Adapts the Branch-and-Price-and-Cut (BPC) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import BPCConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.branch_and_price_and_cut import run_bpc

from .params import BPCParams


@PolicyRegistry.register("bpc")
class BPCPolicy(BaseRoutingPolicy):
    """
    Branch-and-Price-and-Cut policy class.

    Visits pre-selected 'must_go' bins using exact or heuristic BPC solvers.
    """

    def __init__(self, config: Optional[Union[BPCConfig, Dict[str, Any]]] = None):
        """Initialize BPC policy with optional config.

        Args:
            config: BPCConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return BPCConfig

    def _get_config_key(self) -> str:
        """Return config key for BPC."""
        return "bpc"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run BPC solver.

        All nodes in mandatory_nodes are treated as must-go for the solver.
        In VRPP mode, additional nodes from sub_wastes might be collected if profitable.

        Returns:
            Tuple of (routes, profit, solver_cost)
            - routes: List of routes (list of node indices).
            - profit: Objective value (collected revenue - distance cost) in $.
            - solver_cost: Raw travel distance (km), NOT multiplied by cost_unit.
              Callers needing monetary cost should compute solver_cost * cost_unit.
        """
        # Return contract for run_bpc:
        #   routes          — list of customer-node lists (depot excluded)
        #   objective_value — net profit = Σ(revenue_i) - travel_cost, in monetary units.
        #                     May be a greedy-fallback value if BPC found no integer solution.
        # Convert local mandatory indices to a set of must-go nodes for the solver
        must_go_indices: Set[int] = set(mandatory_nodes)

        # Initialize standardized params object (Phase 1 refactoring)
        params = BPCParams.from_config(values)

        # Extract vehicle limit from simulation context (sim.n_vehicles)
        n_vehicles = kwargs.get("n_vehicles")
        # Explicit int conversion and positive check. None and 0 both map to
        # unlimited fleet. False is rejected at the int() call (TypeError surfaced
        # to the caller) rather than silently treated as unlimited.
        vehicle_limit = None if n_vehicles is None else int(n_vehicles) if int(n_vehicles) > 0 else None

        # run_bpc returns (routes, objective_value) where objective_value is
        # net profit (revenue - travel_cost) in the problem's monetary units.
        # It is NOT a raw travel cost despite the variable name used in run_bpc's
        # return signature. Rename immediately to prevent future misreading.
        routes, objective_value = run_bpc(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            must_go_indices=must_go_indices,
            vehicle_limit=vehicle_limit,
            env=kwargs.get("model_env"),
            node_coords=kwargs.get("node_coords"),
            recorder=kwargs.get("recorder"),
        )

        profit = objective_value

        # Compute raw travel distance (km)
        raw_distance = 0.0
        for route in routes:
            # Normalize: strip any leading/trailing depot index before wrapping.
            # Route.nodes stores customer-only sequences, but defensive stripping
            # guards against representation changes in run_bpc's return value.
            inner = [n for n in route if n != 0]
            if not inner:
                continue

            path = [0] + inner + [0]
            for i in range(len(path) - 1):
                raw_distance += sub_dist_matrix[path[i]][path[i + 1]]

        return routes, profit, raw_distance
