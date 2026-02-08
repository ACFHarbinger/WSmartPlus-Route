"""
Base Routing Policy Module.

Provides a template base class for routing policies, extracting common
functionality like parameter loading, subset matrix creation, and tour mapping.

This eliminates code duplication across policy_*.py files.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.interfaces.adapter import IPolicyAdapter
from logic.src.policies.tsp import get_route_cost


class BaseRoutingPolicy(IPolicyAdapter):
    """
    Base class for routing policies with common utilities.

    Subclasses implement `_run_solver()` and optionally `_get_config_key()`.
    The `execute()` method provides a template pattern handling:
        1. Must-go validation
        2. Parameter loading
        3. Subset problem creation
        4. Solver invocation (delegated to subclass)
        5. Tour mapping back to global IDs
        6. Cost computation

    Context Inputs (from kwargs):
        must_go: List of 1-based bin IDs to visit
        bins: Bins state object with `.c` fill levels
        distance_matrix: Full NxN distance matrix
        area: Geographic area name
        waste_type: Waste type (e.g., "plastic")
        config: Policy configuration dict

    Context Outputs:
        Tuple[List[int], float, Any]: (tour, cost, extra_output)

    Attributes:
        config: Optional configuration dataclass for this policy.
    """

    def __init__(self, config: Any = None):
        """Initialize policy with optional config dataclass.

        Args:
            config: Optional dataclass with policy-specific parameters.
        """
        self._config = config

    @property
    def config(self) -> Any:
        """Return the policy configuration dataclass."""
        return self._config

    def _get_config_key(self) -> str:
        """
        Return the config key for this policy (e.g., 'alns', 'bcp', 'hgs').

        Override in subclasses.
        """
        return "default"

    def _validate_must_go(self, must_go: Optional[List[int]]) -> Optional[Tuple[List[int], float, Any]]:
        """
        Validate must_go input. Returns early result if empty.

        Args:
            must_go: List of bin IDs to visit.

        Returns:
            Empty tour tuple if no targets, else None (continue processing).
        """
        if not must_go:
            return [0, 0], 0.0, None
        return None

    def _load_area_params(
        self, area: str, waste_type: str, config: Dict[str, Any]
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Load area-specific parameters (Q, R, C) and merge with config overrides.

        Args:
            area: Geographic area name.
            waste_type: Waste type string.
            config: Policy configuration dict.

        Returns:
            Tuple of (capacity, revenue, cost_unit, merged_values_dict)
        """
        from dataclasses import asdict, is_dataclass

        from logic.src.utils.data.data_utils import load_area_and_waste_type_params

        config_key = self._get_config_key()
        policy_config = config.get(config_key, {})

        # If we have a dataclass config, convert to dict and merge
        if self._config is not None and is_dataclass(self._config):
            policy_config = {**policy_config, **asdict(self._config)}

        Q, R, _, C, _ = load_area_and_waste_type_params(area, waste_type)

        capacity = policy_config.get("capacity", Q)
        revenue = policy_config.get("revenue", R)
        cost_unit = policy_config.get("cost_unit", C)

        values = {"Q": capacity, "R": revenue, "C": cost_unit}
        values.update(policy_config)

        return capacity, revenue, cost_unit, values

    def _create_subset_problem(
        self,
        must_go: List[int],
        distance_matrix: Any,
        bins: Any,
    ) -> Tuple[np.ndarray, Dict[int, float], List[int]]:
        """
        Create a subset distance matrix and demands dict for the solver.

        Args:
            must_go: List of 1-based global bin IDs to visit.
            distance_matrix: Full distance matrix (list or ndarray).
            bins: Bins object with `.c` attribute (0-indexed fill levels).

        Returns:
            Tuple of (sub_dist_matrix, sub_demands, subset_indices)
                - sub_dist_matrix: Distance matrix for subset (depot + must_go)
                - sub_demands: {local_idx: fill} for local nodes 1..M
                - subset_indices: Mapping from local to global indices
        """
        # subset_indices[0] = depot (0), subset_indices[1..M] = must_go bins
        subset_indices = [0] + list(must_go)

        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]

        # Build local demands: local index i (1-based) -> fill level
        sub_demands = {}
        for i, global_idx in enumerate(must_go, 1):
            # global_idx is 1-based, bins.c is 0-indexed
            fill = bins.c[global_idx - 1]
            sub_demands[i] = float(fill)

        return sub_dist_matrix, sub_demands, subset_indices

    def _map_tour_to_global(self, routes: List[List[int]], subset_indices: List[int]) -> List[int]:
        """
        Map solver routes (local indices) back to global bin IDs.

        Args:
            routes: List of routes from solver, each a list of local node indices.
            subset_indices: Mapping from local index to global index.

        Returns:
            Flat tour list with depot (0) separating routes.
        """
        tour = [0]
        if routes:
            for route in routes:
                for node_idx in route:
                    tour.append(subset_indices[node_idx])
                tour.append(0)

        if len(tour) == 1:
            tour = [0, 0]

        return tour

    def _compute_cost(self, distance_matrix: Any, tour: List[int]) -> float:
        """
        Compute total tour cost.

        Args:
            distance_matrix: Full distance matrix.
            tour: Tour as list of node indices.

        Returns:
            Total distance cost.
        """
        return get_route_cost(distance_matrix, tour)

    @abstractmethod
    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_demands: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float]:
        """
        Run the specific solver for this policy.

        Args:
            sub_dist_matrix: Subset distance matrix.
            sub_demands: Local demands dict.
            capacity: Vehicle capacity.
            revenue: Revenue per unit.
            cost_unit: Cost per distance unit.
            values: Merged config values.
            **kwargs: Additional solver-specific arguments.

        Returns:
            Tuple of (routes, solver_cost)
        """
        pass

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the routing policy using template method pattern.

        Args:
            **kwargs: Simulation context.

        Returns:
            Tuple of (tour, cost, extra_output)
        """
        must_go = kwargs.get("must_go", [])

        # 1. Validate
        early_result = self._validate_must_go(must_go)
        if early_result is not None:
            return early_result

        # 2. Extract context
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")

        # 3. Load parameters
        capacity, revenue, cost_unit, values = self._load_area_params(area, waste_type, config)

        # 4. Create subset problem
        sub_dist_matrix, sub_demands, subset_indices = self._create_subset_problem(must_go, distance_matrix, bins)

        # 5. Run solver (subclass-specific)
        routes, _ = self._run_solver(
            sub_dist_matrix,
            sub_demands,
            capacity,
            revenue,
            cost_unit,
            values,
            **kwargs,
        )

        # 6. Map back to global IDs
        tour = self._map_tour_to_global(routes, subset_indices)

        # 7. Compute cost
        cost = self._compute_cost(distance_matrix, tour)

        return tour, cost, None
