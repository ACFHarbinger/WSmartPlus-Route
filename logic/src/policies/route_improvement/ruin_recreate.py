import random
from typing import Any, List

import numpy as np

from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.heuristics.large_neighborhood_search import (
    apply_lns,
)

from .base import RouteImproverRegistry
from .common.helpers import (
    assemble_tour,
    resolve_mandatory_nodes,
    split_tour,
    to_numpy,
    tour_distance,
    upgrade_repair_op_to_profit,
)


@RouteImproverRegistry.register("ruin_recreate")
class RuinRecreateRouteImprover(IRouteImprovement):
    """
    Ruin and Recreate route improver (Large Neighborhood Search).
    Randomly removes a subset of bins and re-inserts them greedily.
    Delegates to operators.heuristics.apply_lns for the ruin-and-recreate pass.
    """

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Apply Ruin and Recreate to the tour.

        Args:
            tour: Initial tour (List of bin IDs including depot 0s).
            **kwargs: Context containing 'distance_matrix', 'lns_iterations',
                     'ruin_fraction', 'lns_acceptance', 'destroy_op', 'repair_op', etc.

        Returns:
            List[int]: Refined tour.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour

        # Parameters
        iterations = kwargs.get("lns_iterations", self.config.get("lns_iterations", 100))
        ruin_fraction = kwargs.get("ruin_fraction", self.config.get("ruin_fraction", 0.2))
        acceptance = kwargs.get("lns_acceptance", self.config.get("lns_acceptance", "best")).lower()
        destroy_op = kwargs.get("destroy_op", self.config.get("destroy_op", "random"))
        repair_op_raw = kwargs.get("repair_op", self.config.get("repair_op", "greedy"))
        repair_k = kwargs.get("repair_k", self.config.get("repair_k", 2))

        # Operator-specific parameters
        p = kwargs.get("p", self.config.get("p", 1.0))
        noise = kwargs.get("noise", self.config.get("noise", 0.0))

        if acceptance not in ("best", "sa"):
            raise ValueError(f"Unknown acceptance rule: {acceptance!r}")

        temp = kwargs.get("lns_sa_temperature", self.config.get("lns_sa_temperature", 1.0))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        cost_per_km = kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0))
        revenue_kg = kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0))

        # Mandatory nodes resolution
        mandatory_nodes = resolve_mandatory_nodes(kwargs, self.config)

        # Profit-aware upgrade
        repair_op = upgrade_repair_op_to_profit(repair_op_raw, revenue_kg, cost_per_km)

        # Convert distance matrix to numpy
        dm = to_numpy(distance_matrix)
        try:
            # We split the RNG streams for independence
            lns_rng = random.Random(seed)
            sa_rng = random.Random(seed + 1)

            current_routes = split_tour(tour)
            if not current_routes:
                return tour

            current_cost = tour_distance(current_routes, dm)

            best_routes = [r[:] for r in current_routes]
            best_cost = current_cost

            # LNS Loop
            for _ in range(iterations):
                # 1 & 2. Ruin and Recreate via apply_lns
                trial_routes = apply_lns(
                    routes=current_routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    R=revenue_kg,
                    C=cost_per_km,
                    rng=lns_rng,
                    ruin_fraction=ruin_fraction,
                    destroy_op=destroy_op,
                    repair_op=repair_op,
                    repair_k=repair_k,
                    mandatory_nodes=mandatory_nodes,
                    p=p,
                    noise=noise,
                )

                # 3. Acceptance
                trial_cost = tour_distance(trial_routes, dm)
                delta = trial_cost - current_cost

                accepted = False
                if acceptance == "best":
                    accepted = trial_cost < current_cost - 1e-6
                else:  # "sa"
                    # For SA, we use the dedicated sa_rng stream
                    accepted = delta < 0 or sa_rng.random() < np.exp(-delta / temp)

                if accepted:
                    current_routes = trial_routes
                    current_cost = trial_cost
                    if current_cost < best_cost - 1e-6:
                        best_cost = current_cost
                        best_routes = [r[:] for r in current_routes]

            return assemble_tour(best_routes)

        except Exception:
            return tour
