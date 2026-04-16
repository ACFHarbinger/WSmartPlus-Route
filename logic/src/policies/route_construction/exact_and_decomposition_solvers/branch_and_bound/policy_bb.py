"""
Branch-and-Bound (BB) Policy Adapter.

Adapts the core mathematical Branch-and-Bound logic to the systems-agnostic
policy interface, handling parameter mapping, profit calculation, and environment
integration.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import BBConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import RouteConstructorRegistry

from .dispatcher import run_bb_optimizer
from .params import BBParams


@RouteConstructorRegistry.register("bb")
class BranchAndBoundPolicy(BaseRoutingPolicy):
    """
    Adapter for the exact Branch-and-Bound routing solver.

    This policy implements a deterministic Best-Bound-First Branch-and-Bound
    search engine based on LP relaxations, mapping high-level simulation
    states (fill levels, distance matrices) into a structured mathematical
    programming model.

    The custom Python implementation of the B&B tree is purposefully designed
    for **full observability of the search state**, enabling the internal
    integration of machine learning components (e.g., neural branching
    heuristics) that are otherwise opaque in closed-source commercial solvers.

    As an exact solver, it guarantees global optimality within the specified
    MIP gap, provided the search terminates within the time limit.
    """

    def __init__(self, config: Optional[Union[BBConfig, Dict[str, Any]]] = None):
        """Initialize the BB policy adapter.

        Args:
            config: A typed BBConfig dataclass, a raw dictionary for parsing,
                or None to use framework defaults.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration dataclass type for automatic parsing."""
        return BBConfig

    def _get_config_key(self) -> str:
        """Return the unique identification key for this policy's configuration."""
        return "bb"

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
        Coordinate the execution of the exact Branch-and-Bound solver.

        Translates the simulation context into the lower-level mathematical
        solver parameters and post-calculates the actual profit returned
        by the found routes.

        The formulation (MTZ or DFJ) is selected based on the configuration:
        - MTZ: Custom B&B with compact formulation
        - DFJ: Gurobi's internal B&B with lazy cuts (default)

        Args:
            sub_dist_matrix: The localized distance matrix for candidates.
            sub_wastes: Current fill levels for available customer nodes.
            capacity: The maximum payload of the vehicle.
            revenue: Expected revenue per unit of waste collected.
            cost_unit: Cost per unit of distance traveled.
            values: Merged configuration dictionary from simulation and YAML.
            mandatory_nodes: Indices of nodes that MUST be visited.

        Returns:
            A tuple containing (List of routes, total profit, total travel cost).
        """
        # Convert local mandatory indices to a set for fast lookup in branching
        mandatory_indices = set(mandatory_nodes)

        # Standardize configuration to BBParams
        params = BBParams.from_config(values)

        # Trigger core solver logic with formulation dispatch
        routes, solver_cost = run_bb_optimizer(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_indices=mandatory_indices,
            env=kwargs.get("model_env"),
            recorder=kwargs.get("recorder"),
        )

        # Internal Profit Calculation: Revenue - Distance Cost
        # This mirrors the objective used during the search for consistency.
        visited = {n for route in routes for n in route}
        collected_revenue = sum(sub_wastes.get(n, 0) * revenue for n in visited)
        dist_cost = 0.0
        for route in routes:
            path = [0] + route + [0]
            for i in range(len(path) - 1):
                dist_cost += sub_dist_matrix[path[i]][path[i + 1]]
        profit = collected_revenue - dist_cost * cost_unit

        monetary_travel_cost = dist_cost * cost_unit
        return routes, profit, monetary_travel_cost
