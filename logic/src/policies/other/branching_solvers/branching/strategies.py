"""
Branching strategies and heuristics for VRPP Column Generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.other.branching_solvers.common.route import Route


class BranchingStrategy(ABC):
    """Abstract base class for finding branching candidates."""

    @abstractmethod
    def find_branching_candidates(self, routes: List[Route], values: Dict[int, float]) -> List[Tuple[float, any]]:
        pass


class EdgeBranching:
    """
    Edge-based branching: select the most-fractional arc and split on it.

    Produces two child nodes:
        left  → x_{uv} = 1  (arc MUST be used)
        right → x_{uv} = 0  (arc is FORBIDDEN)
    """

    @staticmethod
    def find_branching_arc(
        routes: List[Route],
        values: Dict[int, float],
        dist_threshold: float = 0.5,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Identify the arc whose total fractional flow is closest to 0.5.
        """
        arc_flows: Dict[Tuple[int, int], float] = {}
        for r_idx, route in enumerate(routes):
            val = values.get(r_idx, 0.0)
            if val < 1e-6:
                continue

            full_path = [0] + route.nodes + [0]
            for i in range(len(full_path) - 1):
                arc = (full_path[i], full_path[i + 1])
                arc_flows[arc] = arc_flows.get(arc, 0.0) + val

        best_arc: Optional[Tuple[int, int]] = None
        best_dist = 1.0

        for arc, flow in arc_flows.items():
            if 1e-4 < flow < 1 - 1e-4:
                dist = abs(flow - dist_threshold)
                if dist < best_dist:
                    best_dist = dist
                    best_arc = arc

        if best_arc is None:
            return None

        u, v = best_arc
        return u, v, arc_flows[best_arc]


class RyanFosterBranching:
    """Implementation of Ryan-Foster branching for Subset-Row stability."""

    @staticmethod
    def find_branching_pair(
        routes: List[Route],
        values: Dict[int, float],
    ) -> Optional[Tuple[int, int, float]]:
        """
        Identify a pair of nodes {r, s} whose simultaneous visitation
        is closest to 0.5 across the basis.
        """
        pair_flows: Dict[Tuple[int, int], float] = {}

        for r_idx, route in enumerate(routes):
            val = values.get(r_idx, 0.0)
            if val < 1e-6:
                continue

            nodes = sorted(list(route.node_coverage))
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    pair = (nodes[i], nodes[j])
                    pair_flows[pair] = pair_flows.get(pair, 0.0) + val

        best_pair: Optional[Tuple[int, int]] = None
        best_dist = 1.0

        for pair, flow in pair_flows.items():
            if 1e-3 < flow < 1 - 1e-3:
                dist = abs(flow - 0.5)
                if dist < best_dist:
                    best_dist = dist
                    best_pair = pair

        if best_pair is None:
            return None

        r, s = best_pair
        return r, s, pair_flows[best_pair]


class FleetSizeBranching:
    """Branch on the total number of vehicles used."""

    @staticmethod
    def find_branching_value(values: Dict[int, float]) -> Optional[int]:
        sum_val = sum(values.values())
        if abs(sum_val - round(sum_val)) > 1e-4:
            return int(np.floor(sum_val))
        return None


class NodeVisitationBranching:
    """Branch on the binary visitation variable y_i of an optional node."""

    @staticmethod
    def find_branching_node(
        routes: List[Route],
        values: Dict[int, float],
    ) -> Optional[Tuple[int, float]]:
        """Identify node i with visitation sum closest to 0.5."""
        node_visitation: Dict[int, float] = {}

        for r_idx, route in enumerate(routes):
            val = values.get(r_idx, 0.0)
            if val < 1e-6:
                continue
            for node in route.node_coverage:
                node_visitation[node] = node_visitation.get(node, 0.0) + val

        best_node = -1
        best_dist = 1.0

        for node, flow in node_visitation.items():
            if 1e-3 < flow < 1 - 1e-3:
                dist = abs(flow - 0.5)
                if dist < best_dist:
                    best_dist = dist
                    best_node = node

        if best_node == -1:
            return None

        return best_node, node_visitation[best_node]


class MultiEdgePartitionBranching:
    r"""
    Advanced Divergence Branching with Spatial Fleet Partitioning.

    This strategy formalizes the Divergence Branching of Barnhart et al. (1998)
    by utilizing node coordinates to induce a spatially cohesive arc-set
    partition. Unlike ODIMCF which restricts specific vehicles (commodities),
    this restricts the entire anonymous fleet, making it polyhedrally stronger.

    Mathematical Formulation:
    -------------------------
    1. Divergence Node Identification:
       Identify a 'divergence node' $d$ where the fractional flow $\bar{x}$
       diverges into multiple outgoing arcs.
       $A_d^+ = \{(d, v) \in E : 0 < \bar{x}_{dv} < 1\}$

    2. Polar Mapping:
       Define a spatial mapping function $f(v) = \operatorname{atan2}(y_v - y_d, x_v - x_d)$
       returning the polar angle of node $v$ relative to $d$.

    3. Spatial Partitioning:
       Sort $A_d^+$ by destination polar angles and partition into $S_1$ and $S_2$
       via a median split. This creates two balanced geographic sectors.
       $\mathcal{L}: \sum_{(d,v) \in S_1} x_{dv} = 0 \quad \text{and} \quad \mathcal{R}: \sum_{(d,v) \in S_2} x_{dv} = 0$

    4. Candidate Scoring (SVRPC):
       Candidates are ranked by the Spatial Variable Routing Persistence (SVRP)
       strength, calculated as the fractional flow persistence across the split:
       $\sigma(d) = \sum_{(d,v) \in S_1} \bar{x}_{dv}$

    Theoretical Rationale:
    ----------------------
    Spatial fleet partitioning separates the routing topology into convex
    geographic polygons. By forbidding arc sets rather than single edges,
    it globally restricts the anonymous fleet, enforcing a strong bound.
    """

    @staticmethod
    def find_partition_arcs(
        routes: List[Route],
        values: Dict[int, float],
        max_candidates: int = 3,
    ) -> List[Tuple[int, int, float]]:
        arc_flows: Dict[Tuple[int, int], float] = {}
        for r_idx, route in enumerate(routes):
            val = values.get(r_idx, 0.0)
            if val < 1e-6:
                continue

            full_path = [0] + route.nodes + [0]
            for i in range(len(full_path) - 1):
                arc = (full_path[i], full_path[i + 1])
                arc_flows[arc] = arc_flows.get(arc, 0.0) + val

        candidates = []
        for arc, flow in arc_flows.items():
            if 1e-3 < flow < 1 - 1e-3:
                candidates.append((arc[0], arc[1], flow))

        # Sort by distance to 0.5
        candidates.sort(key=lambda x: abs(x[2] - 0.5))
        return candidates[:max_candidates]
