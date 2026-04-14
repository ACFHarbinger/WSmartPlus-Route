"""
Branch-and-Price Post-Processor.

Exact solver for the set-partitioning formulation of the VRP/VRPP, using
column generation (no pool pre-enumeration).

This post-processor delegates to vrpy when available. When vrpy is not
installed, it falls back to a high-perturbation pool-restricted variant
(see SetPartitioningPostProcessor) and logs a warning.

Time budget is critical: branch-and-price can take seconds to minutes
even on small instances, so the post-processor enforces a wall-clock
limit and returns the best feasible solution found within it.
"""

import logging
from typing import Any, List

from logic.src.interfaces.post_processing import IPostProcessor

from .base import PostProcessorRegistry
from .common.helpers import (
    assemble_tour,
    resolve_mandatory_nodes,
    split_tour,
    to_numpy,
    tour_distance,
)

logger = logging.getLogger(__name__)

try:
    from vrpy import VehicleRoutingProblem

    _HAS_VRPY = True
except ImportError:
    _HAS_VRPY = False
    logger.warning(
        "branch_and_price: vrpy not installed; will fall back to "
        "SetPartitioningPostProcessor with sp_n_perturbations=50."
    )


@PostProcessorRegistry.register("branch_and_price")
class BranchAndPricePostProcessor(IPostProcessor):
    """
    Branch-and-price post-processor.

    Solves the set-partitioning formulation of the VRP/VRPP exactly via
    column generation. Requires the `vrpy` package; falls back to a
    high-perturbation set-partitioning variant when vrpy is unavailable.
    """

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:  # noqa: C901
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour

        if not _HAS_VRPY:
            return self._fallback(tour, **kwargs)

        # Solver parameters
        time_limit = kwargs.get("bp_time_limit", self.config.get("bp_time_limit", 120.0))
        cspy = kwargs.get("bp_use_cspy", self.config.get("bp_use_cspy", True))

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        revenue_kg = kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0))
        mandatory_nodes = resolve_mandatory_nodes(kwargs, self.config) or []

        dm = to_numpy(distance_matrix)

        try:
            input_routes = split_tour(tour)
            if not input_routes:
                return tour

            input_cost = tour_distance(input_routes, dm)

            # Build vrpy graph from the distance matrix.
            # vrpy expects a networkx DiGraph with "Source"/"Sink" nodes and
            # edge weights as the "cost" attribute.
            import networkx as nx

            visited_in_input = {n for r in input_routes for n in r}

            # Restrict the graph to bins present in the input + mandatory.
            # This is what makes branch-and-price tractable as a post-processor:
            # we don't reconsider the universe of all bins, only the working set.
            active_nodes = sorted(visited_in_input | set(mandatory_nodes))
            if not active_nodes:
                return tour

            G = nx.DiGraph()
            G.add_node("Source", demand=0)
            G.add_node("Sink", demand=0)
            for n in active_nodes:
                G.add_node(n, demand=int(wastes.get(n, 0)))
                G.add_edge("Source", n, cost=float(dm[0, n]))
                G.add_edge(n, "Sink", cost=float(dm[n, 0]))
            for u in active_nodes:
                for v in active_nodes:
                    if u != v:
                        G.add_edge(u, v, cost=float(dm[u, v]))

            prob = VehicleRoutingProblem(
                G,
                load_capacity=int(capacity) if capacity != float("inf") else None,
            )

            # Profit semantics: vrpy supports prize collection via node
            # "collect" attribute and the prize_collection flag.
            if revenue_kg > 0:
                for n in active_nodes:
                    G.nodes[n]["collect"] = float(wastes.get(n, 0.0)) * revenue_kg
                prob.prize_collection = True

            # Mandatory nodes: vrpy uses a "required" node attribute.
            for m_node in mandatory_nodes:
                if m_node in G.nodes:
                    G.nodes[m_node]["required"] = True

            prob.solve(
                cspy=cspy,
                time_limit=time_limit,
                solver="cbc",  # vrpy auto-detects gurobi/cplex if installed
            )

            if prob.best_routes is None or not prob.best_routes:
                logger.warning("branch_and_price: solver returned no feasible solution; keeping input tour.")
                return tour

            # vrpy returns a dict {route_id: [Source, n1, n2, ..., Sink]}
            refined: List[List[int]] = []
            for _, vrpy_route in prob.best_routes.items():
                stripped = [n for n in vrpy_route if n not in ("Source", "Sink")]
                if stripped:
                    refined.append(stripped)

            # Acceptance gate
            refined_cost = tour_distance(refined, dm)
            if refined_cost > input_cost + 1e-6:
                return tour

            return assemble_tour(refined)

        except Exception as e:
            logger.warning("branch_and_price failed: %s; falling back to set_partitioning.", e)
            return self._fallback(tour, **kwargs)

    def _fallback(self, tour: List[int], **kwargs: Any) -> List[int]:
        """
        Fallback path when vrpy is unavailable or the solver fails.

        Delegates to the pool-restricted set-partitioning post-processor with
        a high perturbation count to compensate for not generating columns
        on demand.
        """
        from .set_partitioning import SetPartitioningPostProcessor

        kwargs.setdefault("sp_n_perturbations", 50)

        sp = SetPartitioningPostProcessor(config=self.config)
        return sp.process(tour, **kwargs)
