"""
Cutting plane separation engines for Branch-and-Price-and-Cut algorithms.

Provides modular separation algorithms for different cut families:
- RCC (Rounded Capacity Cuts): Standard VRP cuts derived from bin-packing requirements.
- SEC (Subtour Elimination Cuts): Enforces connectivity for prize-collecting variants.
- SRI (Subset-Row Inequalities): Tightens set partitioning for subsets (typically size 3).
- LCI (Lifted Cover Inequalities): Heuristic edge-based capacity strengthening.

Methodology:
    Each engine implements a `separate_and_add_cuts` method that inspects the
    fractional LP solution of the Master Problem, identifies violated
    inequalities from its specific family, and registers them into the MP's
    Gurobi model.

References:
    - Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004).
      "A new branch-and-cut algorithm for the capacitated vehicle routing problem."
      Mathematical Programming, 100(2), 423-445.
    - Jepsen, M., Petersen, B., Spoorendonk, S., & Pisinger, D. (2008).
      "Subset-row inequalities applied to the vehicle-routing problem with time windows."
      Operations Research, 56(2), 497-511.
    - Balas, E. (1975). "Facets of the knapsack polytope."
      Mathematical Programming, 8(1), 146-164 (Cover inequalities).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..branch_and_cut.separation import (
    CapacityCut,
    PCSubtourEliminationCut,
    SeparationEngine,
)
from ..branch_and_cut.vrpp_model import VRPPModel
from ..branch_and_price.master_problem import VRPPMasterProblem


class CuttingPlaneEngine(ABC):
    """
    Abstract base class for cutting plane separation engines.

    Each engine implements a specific family of valid inequalities and
    provides separation algorithms to identify violated cuts.
    """

    @abstractmethod
    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Identify violated cuts and add them to the master problem.

        Args:
            master: Master problem instance
            max_cuts: Maximum number of cuts to add
            **kwargs: Engine-specific parameters

        Returns:
            Number of cuts added
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the engine name for logging."""
        pass


class RoundedCapacityCutEngine(CuttingPlaneEngine):
    """
    Rounded Capacity Cut (RCC) separation engine for VRPP Set Packing.

    Mathematical Formulation:
    -------------------------
    For a subset S ⊆ N \\ {0} of customer nodes, the RCC is:

        ∑_{e ∈ δ(S)} x_e ≥ 2 * k(S)

    where:
    - δ(S) is the set of edges crossing the boundary of S
    - k(S) = ⌈demand(S) / Q⌉ is the minimum number of vehicles needed
    - x_e is the aggregated edge usage from the column generation solution

    For VRPP Set Packing, we relax this with visitation variables:

        ∑_{e ∈ δ(S)} x_e ≥ 2*k(S) - M * ∑_{i ∈ S} (1 - y_i)

    where y_i = ∑_{k: i ∈ route_k} λ_k is the fractional visitation probability.

    Integration with Pricing:
    --------------------------
    The dual variables from these cuts must be integrated into the pricing
    problem's reduced cost calculation. For each cut on set S:

        reduced_cost -= dual_S * (number of times route crosses δ(S))

    This is handled by the master problem's dual tracking and passed to
    the RCSPP solver via capacity_cut_duals parameter.

    References:
    -----------
    Lysgaard et al. (2004), Section 3.2: "Capacity cuts"
    """

    def __init__(self, v_model: VRPPModel, sep_engine: SeparationEngine):
        """
        Initialize the RCC separation engine.

        Args:
            v_model: VRPP model for edge indexing and demand tracking
            sep_engine: Separation engine from branch-and-cut module
        """
        self.v_model = v_model
        self.sep_engine = sep_engine

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Separate RCCs and add to master problem.

        Algorithm:
        1. Get current edge usage from master problem's LP solution
        2. Map edge usage to the separation engine's representation
        3. Run separation algorithms (heuristic + exact)
        4. Add violated cuts to master with Set Packing relaxation

        Args:
            master: Master problem instance
            max_cuts: Maximum number of cuts to add
            **kwargs: Unused (for interface consistency)

        Returns:
            Number of cuts added
        """
        edge_vars = master.get_edge_usage()
        if not edge_vars:
            return 0

        # Map edge usage to x_vals array for SeparationEngine
        x_vals = np.zeros(len(self.v_model.edges))
        for (i, j), val in edge_vars.items():
            edge_tuple = (min(i, j), max(i, j))
            if edge_tuple in self.v_model.edge_to_idx:
                idx = self.v_model.edge_to_idx[edge_tuple]
                x_vals[idx] = val

        # Build y_vals: fractional node visitation probabilities from the LP solution.
        # y_vals[i-1] = Σ_{k: i ∈ route_k} λ_k for each customer node i.
        # Index 0 corresponds to customer node 1 (depot node 0 is excluded).
        node_visits = master.get_node_visitation()
        n_customers = self.v_model.n_nodes - 1  # exclude depot
        y_vals = np.zeros(n_customers)
        for node, val in node_visits.items():
            if 1 <= node <= n_customers:
                y_vals[node - 1] = min(val, 1.0)  # clamp to [0,1] for LP fractional values

        # Separate cuts using the existing separation engine
        # BPC separation always operates on fractional LP solutions.
        # Use separate_fractional() for exact max-flow based RCC separation.
        violated_ineqs = self.sep_engine.separate_fractional(x_vals, y_vals=y_vals, max_cuts=max_cuts)
        added_cuts = 0

        for ineq in violated_ineqs:
            if isinstance(ineq, CapacityCut):
                node_set = list(ineq.node_set)

                # For Set Packing, add cut with boundary relaxation
                # The master problem handles the y_i visitation tracking
                if master.add_set_packing_capacity_cut(node_set, ineq.rhs):
                    added_cuts += 1
            elif isinstance(ineq, PCSubtourEliminationCut):
                node_set = list(ineq.node_set)
                is_global = ineq.facet_form == "2.1"
                if master.add_sec_cut(node_set, ineq.rhs, cut_name=ineq.facet_form, global_cut=is_global):
                    added_cuts += 1

        return added_cuts

    def get_name(self) -> str:
        return "rcc"


class SubsetRowCutEngine(CuttingPlaneEngine):
    """
    Subset-Row Inequality (SRI) separation engine.

    Specifically implements 3-SRI (k=3, p=2):
    Σ_{k} ⌊ 1/2 * |S ∩ route_k| ⌋ * λ_k  <= 1

    Heuristic Separation:
    1. Identify customer nodes with high fractional visitation.
    2. Check subsets of 3 nodes that are spatially close.
    3. Evaluate violation and add to master.
    """

    def __init__(self, v_model: VRPPModel):
        self.v_model = v_model

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        if master.model is None or len(master.routes) < 2:
            return 0

        # Heuristic: only look at nodes with 0.1 < visitation < 0.9
        visitation = master.get_node_visitation()
        candidates = [n for n, v in visitation.items() if 0.1 < v < 0.9 and n != 0]

        if len(candidates) < 3:
            return 0

        # Sort candidates by visitation decr
        candidates.sort(key=lambda n: visitation[n], reverse=True)
        top_candidates = candidates[:15]  # Limit search space

        added = 0
        import itertools

        for subset in itertools.combinations(top_candidates, 3):
            if added >= max_cuts:
                break

            val = 0.0
            node_set = set(subset)
            for idx, route in enumerate(master.routes):
                lam = master.lambda_vars[idx].X
                if lam < 1e-6:
                    continue
                count = len(node_set.intersection(route.node_coverage))
                coeff = count // 2
                if coeff > 0:
                    val += float(coeff) * lam

            if val > 1.0 + 1e-4 and master.add_subset_row_cut(list(subset)):
                added += 1

        return added

    def get_name(self) -> str:
        return "sri"


class LiftedCoverCutEngine(CuttingPlaneEngine):
    """
    Lifted Cover Inequalities (LCI) for VRPP.

    Implemented specifically as tightened edge-capacity clique cuts:
        Σ_{k: {u,v} ⊆ Route_k} λ_k <= 1

    Heuristic Separation:
        Inspects the current fractional flow x_{uv} = Σ_{k} δ_{uv,k} * λ_k.
        If x_{uv} > 1.0 + epsilon, the edge capacity is violated.
        Because each physical edge can accommodate at most one vehicle,
        the sum of fractional routes traversing edge (u,v) must be <= 1.0.
    """

    def __init__(self, v_model: VRPPModel, epsilon: float = 0.01) -> None:
        self.v_model = v_model
        self.epsilon = epsilon

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Identify edges where fractional flow exceeds 1.0.
        """
        # Calculate edge flow: x_e = Σ_{k: e ∈ k} λ_k
        route_values = {i: var.X for i, var in enumerate(master.lambda_vars) if var.X > 1e-6}
        edge_flow: Dict[Tuple[int, int], float] = {}
        for idx, val in route_values.items():
            route = master.routes[idx]
            nodes = [0] + route.nodes + [0]
            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]
                # Exclude edges involving the depot (node 0)
                if u == 0 or v == 0:
                    continue

                e = tuple(sorted((u, v)))
                edge_flow[e] = edge_flow.get(e, 0.0) + val  # type: ignore[index,arg-type]

        count = 0
        for edge_tuple, flow in edge_flow.items():
            if count >= max_cuts:
                break
            if flow > 1.0 + self.epsilon:
                u, v = edge_tuple
                if master.add_edge_lci_cut(u, v):
                    count += 1
        return count

    def get_name(self) -> str:
        return "lci"


class KnapsackCoverEngine(CuttingPlaneEngine):
    """
    Separates Cover Inequalities for knapsack constraints in the Master Problem.

    In the VRPP context, this primarily targets the fleet size (vehicle limit)
    constraint: Σ λ_k ≤ K. While simpler than a general knapsack with varied
    weights, separating covers for the fleet limit helps strengthen the
    integrality of the master problem in tightly constrained instances.

    Reference: Barnhart et al. (2000), Section 5.

    Interaction with the fleet constraint:
        When VRPPMasterProblem has an active vehicle_limit constraint
        (Σ λ_k ≤ K), the fleet constraint already prevents the LP from
        selecting more than K routes. In that case, this engine's cover cut
        (Σ_{k ∈ top_K+1} λ_k ≤ K) is redundant and will never fire.

        This engine is primarily useful when NO explicit vehicle_limit is set
        in the master problem — e.g., in uncapacitated-fleet VRPP variants —
        where the cover cut provides a surrogate fleet-size tightening.

        If vehicle_limit is always set, consider disabling this engine to
        avoid unnecessary separation overhead.
    """

    def __init__(self, v_model: VRPPModel, sep_engine: SeparationEngine):
        self.v_model = v_model
        self.sep_engine = sep_engine

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        """
        Identify and add violated cover inequalities for the vehicle limit.
        """
        if master.vehicle_limit is None or master.model is None:
            return 0

        # Σ λ_k ≤ K.
        # A cover C is any subset of routes such that |C| > K.
        # The inequality is Σ_{k ∈ C} λ_k ≤ |C| - 1.
        # Violation occurs if Σ_{k ∈ C} λ_k > |C| - 1.

        # Strategy: find routes with largest fractional values.
        # If the sum of the top K+1 λ_k values exceeds K, we have a violation.
        route_values = {i: var.X for i, var in enumerate(master.lambda_vars) if var.X > 1e-6}
        if len(route_values) <= master.vehicle_limit:
            return 0

        sorted_indices = sorted(route_values.keys(), key=lambda i: route_values[i], reverse=True)
        top_k_plus_1 = sorted_indices[: master.vehicle_limit + 1]
        sum_top = sum(route_values[i] for i in top_k_plus_1)

        if sum_top > master.vehicle_limit + 1e-4:
            # Violated cover found.
            # The cover is the set of nodes visited by the top K+1 routes.
            # Use their union as the node set S for the capacity cut registry.
            cover_node_set: set = set()
            for i in top_k_plus_1:
                cover_node_set.update(master.routes[i].node_coverage)
            cover_node_set.discard(0)  # exclude depot

            if cover_node_set and master.add_set_packing_capacity_cut(
                list(cover_node_set), float(master.vehicle_limit)
            ):
                return 1

        return 0

    def get_name(self) -> str:
        return "cover"


class CompositeCuttingPlaneEngine(CuttingPlaneEngine):
    """
    Combines multiple separation engines to run in sequence.
    """

    def __init__(self, engines: List[CuttingPlaneEngine]) -> None:
        self.engines = engines

    def separate_and_add_cuts(self, master: VRPPMasterProblem, max_cuts: int, **kwargs) -> int:
        total_added = 0
        for engine in self.engines:
            if total_added >= max_cuts:
                break
            total_added += engine.separate_and_add_cuts(master, max_cuts - total_added)
        return total_added

    def get_name(self) -> str:
        return "composite"


def create_cutting_plane_engine(
    engine_name: str, v_model: VRPPModel, sep_engine: Optional[SeparationEngine] = None
) -> CuttingPlaneEngine:
    """
    Factory function to create cutting plane engines.

    Args:
        engine_name: Name of the engine ("rcc", "src", "cover", or "all")
        v_model: VRPP model for problem data
        sep_engine: Separation engine (required for RCC/SRC)

    Returns:
        Instance of the requested cutting plane engine

    Raises:
        ValueError: If engine_name is not recognized or required args missing

    Example:
        >>> sep_engine = SeparationEngine(v_model)
        >>> cut_engine = create_cutting_plane_engine("rcc", v_model, sep_engine)
        >>> cuts_added = cut_engine.separate_and_add_cuts(master, max_cuts=5)
    """
    if engine_name == "rcc":
        if sep_engine is None:
            raise ValueError("RCC engine requires sep_engine parameter")
        return RoundedCapacityCutEngine(v_model, sep_engine)
    elif engine_name == "sri":
        return SubsetRowCutEngine(v_model)
    elif engine_name == "lci":
        return LiftedCoverCutEngine(v_model)
    elif engine_name == "cover":
        if sep_engine is None:
            raise ValueError("Cover engine requires sep_engine parameter")
        return KnapsackCoverEngine(v_model, sep_engine)
    elif engine_name in ("all", "composite"):
        engines = []
        if sep_engine is not None:
            engines.append(RoundedCapacityCutEngine(v_model, sep_engine))
        engines.append(SubsetRowCutEngine(v_model))  # type: ignore[arg-type]
        engines.append(LiftedCoverCutEngine(v_model))  # type: ignore[arg-type]
        if sep_engine is not None:
            engines.append(KnapsackCoverEngine(v_model, sep_engine))  # type: ignore[arg-type]
        return CompositeCuttingPlaneEngine(engines)  # type: ignore[arg-type]
    else:
        valid = ["rcc", "sri", "lci", "cover", "all"]
        raise ValueError(f"Unknown cutting plane engine '{engine_name}'. Valid options are: {valid}")
