"""Classical Local Search Route Improver.

This module provides a unified interface for classical local search operators,
supporting both simple descent and more complex iterative improvement methods.

Attributes:
    ClassicalLocalSearchRouteImprover: Unified local search improver class.

Example:
    >>> improver = ClassicalLocalSearchRouteImprover()
    >>> best_tour, metrics = improver.process(tour, distance_matrix=dm, ls_operator="2opt")
"""

from typing import Any, List, Tuple

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces import IRouteImprovement
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.policies.helpers.operators.intensification_fixing import INTENSIFICATION_OPERATORS

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@GlobalRegistry.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.HEURISTIC,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.TRAJECTORY_BASED,
)
@RouteImproverRegistry.register("classical_local_search")
class ClassicalLocalSearchRouteImprover(IRouteImprovement):
    """Wrapper for classical local search operators.

    Drives a multi-route tour to a local minimum using steepest descent or
    iterative improvement. Supports multiple operators like 2-opt, 3-opt,
    swap, and more.

    Attributes:
        config (Dict[str, Any]): Configuration parameters.

    Example:
        >>> improver = ClassicalLocalSearchRouteImprover()
        >>> tour, metrics = improver.process(tour, distance_matrix=dm, ls_operator="2opt")
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """Apply classical local search to the tour.

        Args:
            tour (List[int]): The initial tour sequence.
            kwargs (Any): Search context, including:
                - distance_matrix (np.ndarray): The distance matrix.
                - ls_operator (str): Name of the operator to use (e.g., '2opt', '3opt', 'swap').
                - iterations (int): Maximum number of iterations.
                - wastes (Dict[int, float]): Bin waste demands.
                - capacity (float): Vehicle capacity.
                - R (float): Revenue per kg.
                - C (float): Cost per km.
                - seed (int): Random seed.

        Returns:
            Tuple[List[int], ImprovementMetrics]: Refined tour and performance metrics.
        """

        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None:
            return tour, {"algorithm": "ClassicalLocalSearchRouteImprover"}

        # Get parameters from config (via kwargs) with fallbacks
        max_iter = kwargs.get("iterations", kwargs.get("n_iterations", self.config.get("iterations", 500)))
        operator_name = kwargs.get("ls_operator", kwargs.get("operator_name", self.config.get("ls_operator", "2opt")))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        R = kwargs.get("R", self.config.get("R", 1.0))
        C = kwargs.get("C", self.config.get("C", 1.0))

        dist_matrix = to_numpy(distance_matrix)

        if len(tour) < 3:
            return tour, {"algorithm": "ClassicalLocalSearchRouteImprover"}

        routes = split_tour(tour)
        if not routes:
            return tour, {"algorithm": "ClassicalLocalSearchRouteImprover"}

        # Case 1: Mathematical programming and exact intensification solvers
        Mapping = {
            "dp_reopt": "DP_REOPT_PROFIT",
            "fix_opt": "FIX_OPT_PROFIT",
            "sp_polish": "SP_POLISH_PROFIT",
        }

        op_key = Mapping.get(operator_name.lower())
        if op_key and op_key in INTENSIFICATION_OPERATORS:
            op_fn = INTENSIFICATION_OPERATORS[op_key]
            try:
                refined_routes = op_fn(routes, dist_matrix, wastes, capacity, R=R, C=C, max_iter=max_iter)  # type: ignore[operator]
                return assemble_tour(refined_routes), {"algorithm": "ClassicalLocalSearchRouteImprover"}
            except Exception:
                return tour, {"algorithm": "ClassicalLocalSearchRouteImprover"}

        # Case 2: Multi-operator sequential search loop (from simplest to most complex)
        from logic.src.policies.helpers.local_search.local_search_manager import LocalSearchManager

        manager = LocalSearchManager(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            improvement_threshold=1e-6,
            seed=seed,
        )
        manager.set_routes(routes)

        # Multi-operator list ordered from simplest (cheapest) to most complex (expensive)
        # 1. relocate: Shift single node (simplest O(N) or O(N^2))
        # 2. swap: Exchange two nodes (O(N^2))
        # 3. two_opt_intra: Intra-route segment reversal (O(N^2))
        # 4. or_opt_2: Relocate chain of 2 nodes (O(N^2))
        # 5. or_opt_3: Relocate chain of 3 nodes (O(N^2))
        # 6. two_opt_star: Inter-route tail swap (O(N^2))
        # 7. swap_star: Inter-route node exchange with best insertion (O(N^2))
        # 8. three_opt_intra: Intra-route 3-opt (O(N^3))
        # 9. four_opt_intra: Intra-route 4-opt (O(N^4))
        operators = [
            manager.relocate,
            manager.swap,
            manager.two_opt_intra,
            lambda: manager.or_opt(chain_len=2),
            lambda: manager.or_opt(chain_len=3),
            manager.two_opt_star,
            manager.swap_star,
            manager.three_opt_intra,
            manager.four_opt_intra,
        ]

        try:
            for _ in range(max_iter):
                improved = False
                for op_func in operators:
                    if op_func():
                        improved = True
                        break  # restart search from simplest operator
                if not improved:
                    break  # local optimum with respect to all operators reached
            return assemble_tour(manager.get_routes()), {"algorithm": "ClassicalLocalSearchRouteImprover"}
        except Exception:
            return tour, {"algorithm": "ClassicalLocalSearchRouteImprover"}
